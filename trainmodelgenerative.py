import tensorflow as tf
from tensorflow import keras
import redditdata as rd
import redditmodelgenerative as rmg
import json
import numpy as np
import os
import pickle

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

commentchunks = [list(e) for e in np.array_split(np.array(highlyvotedcomments),10)]
textchunks = []
for commentchunk in commentchunks:
    textchunk = ""
    for comment in commentchunk:
        textchunk += "\n" + comment['text']
    textchunks += [textchunk]

vocab = sorted(set(text))

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

print("Saved idx2char to disk")

f = open('settings/vocab.pckl', 'wb')
pickle.dump(vocab, f)
f.close()

print("Saved vocab to disk")

model.summary()

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = rmg.loss)

for textchunk in textchunks:
    text_as_int = np.array([char2idx[c] for c in textchunk])

    examples_per_epoch = len(textchunk)//rmg.seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(rmg.seq_length+1, drop_remainder=True)

    dataset = sequences.map(rmg.split_input_target)

    dataset = dataset.shuffle(rmg.BUFFER_SIZE).batch(rmg.BATCH_SIZE, drop_remainder=True)

    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(rmg.checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    steps_per_epoch = examples_per_epoch//rmg.BATCH_SIZE

    history = model.fit(dataset.repeat(), epochs=rmg.EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
