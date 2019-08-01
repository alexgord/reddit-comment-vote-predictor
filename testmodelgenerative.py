import tensorflow as tf
from tensorflow import keras
import redditmodelgenerative as rmg
import numpy as np
import pickle

f = open('settings/char2idx.pckl', 'rb')
char2idx = pickle.load(f)
f.close()

print("Read char2idx from disk")

f = open('settings/idx2char.pckl', 'rb')
idx2char = pickle.load(f)
f.close()

print("Read idx2char from disk")

f = open('settings/vocab.pckl', 'rb')
vocab = pickle.load(f)
f.close()

print("Read vocab from disk")

# Length of the vocabulary in chars
vocab_size = len(vocab)

model = rmg.getmodel(vocab_size = vocab_size, embedding_dim=rmg.embedding_dim, rnn_units=rmg.rnn_units, batch_size=rmg.BATCH_SIZE)

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

model = model = rmg.getmodel(vocab_size = vocab_size, embedding_dim=rmg.embedding_dim, rnn_units=rmg.rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(rmg.checkpoint_dir))

model.build(tf.TensorShape([1, None]))

print("\nPredicted text:\n")

print(generate_text(model, start_string=u"Why "))
