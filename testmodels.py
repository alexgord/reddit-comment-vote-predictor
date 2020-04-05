import datetime
import time, threading
import logging
import redditmodel as rm
import redditmodelgenerative as rmg
import redditmodelscience as rms
import redditdata as rd
import tensorflow as tf
import pickle
import praw
from praw.models import MoreComments
from pymongo import MongoClient
import sys

SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * MINUTES_IN_HOUR

#mongodb setup
connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]
client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]
collection = db.comments

#Mode which predicts votes
model = rm.getmodelandweights()

#Get information to build model which generates text
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

#Model which generates text
modelgenerative = rmg.getmodel(vocab_size = vocab_size, embedding_dim=rmg.embedding_dim, rnn_units=rmg.rnn_units, batch_size=1)

modelgenerative.load_weights(rmg.checkpoint_dir)

modelgenerative.build(tf.TensorShape([1, None]))

#Model which predicts if a comment will be removed on /r/science
modelscience = rms.getmodelandweights()

commentstoremove = []
obtainedcommentstoremovetime = None

predictions = rm.getprediction(model, ["title"], [111111], [1], ["text"], collection)

print("Predictions:")
print(predictions)

predictions_science = rms.getprediction(modelscience, ["title"], ["text"])

print("Science predictions:")
print(predictions_science)

generatedtext = rmg.generatesentence(modelgenerative, "This ", char2idx, idx2char)

print("Generated text:")
print(generatedtext)
