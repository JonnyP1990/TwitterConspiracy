# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 12:10:31 2022

@author: jonpr
"""

#### AntiVax Tweet Generator

import os
import tensorflow as tf
import numpy as np
import pickle

# paths
os.chdir(r"C:\Users\jonpr\Documents\Data projects\Python\TwitterConspiracy")
path = os.getcwd()
modelPath = os.path.join(path, r"\save_models\WICO_5G")

idx2char = np.load('idx2char.npy')
f = open("char2idx.pkl","rb")
char2idx = pickle.load(f)
f.close()

# load saved model
model = tf.keras.models.load_model('saved_models/WICO_5G')

# Tweet generator
def generate_tweet(model, start_string):
    num_generate=280 # how much text to generate
    
    input_eval = [char2idx[s] for s in start_string] # vectorize (convert start string to numbers)
    input_eval = tf.expand_dims(input_eval, 0)
    
    text_generated = []
    temperature = 0.8 # low temp = predictible, high temp = surprising
    
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        # use categorical distrubution to predict next character
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        # add the predicted character to the input, before using the new input to predict the next one
        input_eval = tf.expand_dims([predicted_id], 0)
        
        # add the new character to the list
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))
        
inp = input('Type a string to start: ')
print(generate_tweet(model, inp))

        