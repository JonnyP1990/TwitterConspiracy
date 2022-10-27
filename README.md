# TwitterConspiracy

This model generates novel tweets based on a database of anti-vaccination conspiracy tweets

## Dataset
I used the WICO (Wireless Networks and Coronovirus Conspiracy) dataset, which contained approximately 3000 tweets that were labeled as conspiracy-related COVID-19 misinformation.

The full dataset is available here: https://datasets.simula.no/wico-graph/
The related paper can be found here: https://doi.org/10.31219/osf.io/r2t56

## Model
The model uses a particular type of recurrent neural network (RNN) known as a Long Short-Term Memory (LSTM) network. RNN's include an internal memory state to better process the context of incoming data, which is useful for natural language processing.

The model was trained for 100 epochs, and can be loaded from the 'saved_models/' folder. The final training checkpoint can be located in 'training_checkpoints/', and can be used for further fine-tuning of the model.

Non ASCII characters were removed from the dataset to prevent the generation of strings of emojis, so punctuation of the generated tweets needs to be inferred.

## Generating Tweets
Running 'AntiVax_Gen' will prompt you to imput a starting string. The script will then use the model to 'complete' your tweet based on what it has learned from the WICO database.

The 'surprisingness' of generated outputs can be modified by changing the 'temperature' variable (low = more predictable, high = more surprising).

Some examples of cherry-picked tweets can be found in 'WICO_Tweet_Examples.docx'

## Credit
Code used in this project was adapted from:
tensorflow: https://www.tensorflow.org/text/tutorials/text_generation
freecodecamp: https://www.freecodecamp.org/