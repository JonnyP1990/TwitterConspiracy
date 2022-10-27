# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:06:02 2022

@author: jonpr
"""

#### Train model

import os
import re
import numpy as np
import tensorflow as tf
import pickle

# paths
os.chdir(r"C:\Users\jonpr\Documents\Data projects\Python\TwitterConspiracy")
path = os.getcwd()

# Open tweet dataset
textPath = os.path.join(path, "tweetText_WICO.txt")
text = open(textPath, 'rb').read().decode(encoding='utf-8') 

# Remove symbols and non ascII characters from text
text = re.sub(r'[^\w]', ' ', text)
t_encode = text.encode("ascii", "ignore")
text = t_encode.decode()

# Encode characters as integers
vocab = sorted(set(text)) 
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

f = open("char2idx.pkl","wb")
pickle.dump(char2idx, f)
f.close()
np.save('idx2char',idx2char)

# text-int conversion functions
def text_to_int(text):
    return np.array([char2idx[c] for c in text])

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])
            
text_as_int = text_to_int(text)

# ML Parameters
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

# Create training data
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# define a function to build models
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Create a model and pass in a random batch to get an example output
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, '# (batch_size, sequence_length, vocab_size)')

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

# Designing a loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Train
history = model.fit(data, epochs=100, callbacks=[checkpoint_callback])

# Use a checkpoint to rebuild model with a batch size of 1
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Save the model
model.save('saved_models/WICO_5G')