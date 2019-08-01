import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import tensorflow_hub as hub
import redditdata as rd
import json

if tf.test.is_gpu_available():
  rnn = keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    keras.layers.GRU, recurrent_activation='sigmoid')

HIGHLY_VOTED_COMMENT_MINIMUM = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS=30

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# The maximum length sentence we want for a single input in characters
seq_length = 100

# Directory where the checkpoints will be saved
checkpoint_dir = './generator_training_checkpoints'

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def loss(labels, logits):
  return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def getmodel(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        stateful=True),
    keras.layers.Dense(vocab_size)
    ])
    return model
