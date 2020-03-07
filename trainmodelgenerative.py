import tensorflow as tf
from tensorflow import keras
import redditdata as rd
import redditmodelgenerative as rmg
import json
import numpy as np
import math
import os
import pickle
import random

trainpercentage = 0.85
validationpercentage = 1 - trainpercentage

totalcomments = 0
highlyvotedcomments = []
for subreddit_name in rd.subreddit_list:
    comments_file = 'data/comments_' + subreddit_name + '.json'
    with open(comments_file, 'r') as f:
        newcomments = json.load(f)
        totalcomments += len(newcomments)
        highlyvotedcomments += [comment for comment in newcomments if comment['score'] > rmg.HIGHLY_VOTED_COMMENT_MINIMUM]

print("Number of comments: {}".format(totalcomments))
print("Number of comments with score at least {}: {}".format(rmg.HIGHLY_VOTED_COMMENT_MINIMUM, len(highlyvotedcomments)))
print("{}% of comments have score at least {}".format(len(highlyvotedcomments)/totalcomments*100, rmg.HIGHLY_VOTED_COMMENT_MINIMUM))

text = ""
for comment in highlyvotedcomments:
    text += "\n" + comment['text']

vocab = sorted(set(text))
text = ""

# Length of the vocabulary in chars
vocab_size = len(vocab)

model = rmg.getmodel(vocab_size = vocab_size, embedding_dim=rmg.embedding_dim, rnn_units=rmg.rnn_units, batch_size=rmg.BATCH_SIZE)

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

f = open('settings/char2idx.pckl', 'wb')
pickle.dump(char2idx, f)
f.close()

print("Saved char2idx to disk")

f = open('settings/idx2char.pckl', 'wb')
pickle.dump(idx2char, f)
f.close()

idx2char = []

print("Saved idx2char to disk")

f = open('settings/vocab.pckl', 'wb')
pickle.dump(vocab, f)
f.close()

vocab = []

print("Saved vocab to disk")

model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = rmg.loss)

highlyvotedcomments = []
for subreddit_name in rd.subreddit_list:
    print("Gathering comments from {}".format(subreddit_name))
    comments_file = 'data/comments_' + subreddit_name + '.json'
    with open(comments_file, 'r') as f:
        newcomments = json.load(f)
        highlyvotedcomments += [comment for comment in newcomments if comment['score'] > rmg.HIGHLY_VOTED_COMMENT_MINIMUM]
        newcomments = []

random.shuffle(highlyvotedcomments)

highly_voted_comment_texts = [c['text'] for c in highlyvotedcomments]

comments_text_train = highly_voted_comment_texts[:math.floor(len(highly_voted_comment_texts)*trainpercentage)]
comments_text_test = highly_voted_comment_texts[math.ceil(len(highly_voted_comment_texts)*validationpercentage):]

bigcomment_text_train = ""
for comment in comments_text_train:
    bigcomment_text_train += "\n" + comment

bigcomment_text_test = ""
for comment in comments_text_test:
    bigcomment_text_test += "\n" + comment

highlyvotedcomments = []

text_as_int_train = np.array([char2idx[c] for c in bigcomment_text_train])
text_as_int_test = np.array([char2idx[c] for c in bigcomment_text_test])

examples_per_epoch = len(bigcomment_text_train)//rmg.seq_length

# Create training examples / targets
char_dataset_train = tf.data.Dataset.from_tensor_slices(text_as_int_train)

char_dataset_test = tf.data.Dataset.from_tensor_slices(text_as_int_test)

sequences_train = char_dataset_train.batch(rmg.seq_length+1, drop_remainder=True)

sequences_test = char_dataset_train.batch(rmg.seq_length+1, drop_remainder=True)

dataset_train = sequences_train.map(rmg.split_input_target)

dataset_train = dataset_train.shuffle(rmg.BUFFER_SIZE).batch(rmg.BATCH_SIZE, drop_remainder=True)

dataset_test = sequences_test.map(rmg.split_input_target)

dataset_test = dataset_test.shuffle(rmg.BUFFER_SIZE).batch(rmg.BATCH_SIZE, drop_remainder=True)

steps_per_epoch = examples_per_epoch//rmg.BATCH_SIZE

text_as_int = []
char_dataset = []
sequences = []

history = model.fit(dataset_train.repeat(), epochs=rmg.EPOCHS, steps_per_epoch=steps_per_epoch, validation_data = dataset_test)

#Test the model and print results
print("Model evaluation:")
result = model.evaluate(dataset_test, verbose=0)
print("%s: %.3f" % (model.metrics_names[0], result))

model.save_weights(rmg.checkpoint_dir)
