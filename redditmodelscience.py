import tensorflow as tf
from tensorflow import keras
import praw
from praw.models import MoreComments
from datetime import datetime
import tensorflow_hub as hub
import redditdata as rd
import json
import numpy as np

checkpoint_dir = './science_training_checkpoints/my_checkpoint'

def getmodel():
    dropoutrate = 0.3

    hub_layer1 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer2 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)

    # the first branch opreates on the first input
    # This is the post's title
    y = keras.Sequential()
    y.add(hub_layer1)

    # the second branch opreates on the second input
    # This is the comment's text contents, in plaintext
    z = keras.Sequential()
    z.add(hub_layer2)

    # combine the output of the two branches
    combined = keras.layers.concatenate([y.output, z.output])

    # Build the intermediate layers
    zz = keras.layers.Dense(80, activation="relu")(combined)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(80, activation="relu")(zz)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(2, activation="softmax")(zz)

    # our model will accept the inputs of the three branches and
    # then output a single value
    model = keras.Model(inputs=[y.input, z.input], outputs=zz)

    model.summary()

    optimizer = keras.optimizers.Adam()

    model.compile(loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    
    return model

def getmodelandweights():
    model = getmodel()

    model.load_weights(checkpoint_dir)

    return model

def getprediction(model, titles, texts):
    #Build data pipeline
    dataset_titles = tf.data.Dataset.from_tensors(titles)
    dataset_texts = tf.data.Dataset.from_tensors(texts)
    dataset_inputs = tf.data.Dataset.zip((dataset_titles, dataset_texts))
    dataset = tf.data.Dataset.zip((dataset_inputs,))

    return [np.argmax(prediction) for prediction in  model.predict(dataset)]

def getpredictedremovedcomments(model):
    print("Scraping comments from science subreddit")
    appId = ''
    appSecret = ''

    app_settings_file = 'settings/app.json'
    with open(app_settings_file, 'r') as f:
        settings = json.load(f)
        appId = settings['app_id']
        appSecret = settings['secret']

    reddit = praw.Reddit(client_id=appId, client_secret=appSecret, user_agent=rd.user_agent)

    subreddit_name = "science"

    comments = []
    titles = []
    texts = []

    for submission in reddit.subreddit(subreddit_name).new(limit = 100):
        # Removes all MoreComment objects from submission object
        submission.comments.replace_more(limit = None)
        for comment in submission.comments.list():
            comments +=  [rd.extractInfoFromComment(comment, submission, subreddit_name)]
            titles += [submission.title]
            texts += [comment.body]

    predictions = getprediction(model, titles, texts)

    for comment, prediction in zip(comments, predictions):
        comment.update( {"removal_prediction":bool(prediction)})

    commentstoremove = [comment for comment in comments if comment['removal_prediction']
                            and comment['text'] != "[removed]"]

    print("Found all comments")

    return commentstoremove
    