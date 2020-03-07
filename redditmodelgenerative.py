import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import tensorflow_hub as hub
import redditdata as rd
import json

HIGHLY_VOTED_COMMENT_MINIMUM = 50
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS=23

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# The maximum length sentence we want for a single input in characters
seq_length = 100

# Directory where the checkpoints will be saved
checkpoint_dir = './generator_training_checkpoints/my_checkpoint'

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def loss(labels, logits):
  return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def getmodel(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model

def generatesentence(model, start_string, char2idx, idx2char):
  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures result in more predictable text.
  # Higher temperatures result in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  endsentence = ['.','!','?']

  # Here batch size == 1
  model.reset_states()
  previouschar = ''
  while previouschar not in endsentence:
      predictions = model.predict(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      previouschar = idx2char[predicted_id]

      text_generated.append(previouschar)

  return (start_string + ''.join(text_generated))
