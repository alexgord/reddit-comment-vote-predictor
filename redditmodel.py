import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import tensorflow_hub as hub
import redditdata as rd
import json
import math

def getmodel():
    dropoutrate = 0.06

    hub_layer1 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer2 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer3 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer4 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer5 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer6 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)
    hub_layer7 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)

    # the first branch operates on the first input
    # This is the context of the comment, namely:
    # minute, hour, day, month, weekday, subreddit
    inputA = keras.Input(shape=(6,))
    x = keras.layers.Dense(6, activation="relu")(inputA)
    x = keras.Model(inputs=inputA, outputs=x)

    # the second branch opreates on the second input
    # This is the post's title
    y = keras.Sequential()
    y.add(hub_layer1)

    # the third branch opreates on the third input
    # This is the comment's text contents, in plaintext
    z = keras.Sequential()
    z.add(hub_layer2)
    
    # the fourth branch opreates on the third input
    # This is the first of the comment's highly voted comment context
    z1 = keras.Sequential()
    z1.add(hub_layer3)

    # the fifth branch opreates on the third input
    # This is the second of the comment's highly voted comment context
    z2 = keras.Sequential()
    z2.add(hub_layer4)

    # the sixth branch opreates on the third input
    # This is the third of the comment's highly voted comment context
    z3 = keras.Sequential()
    z3.add(hub_layer5)

    # the seventh branch opreates on the third input
    # This is the fourth of the comment's highly voted comment context
    z4 = keras.Sequential()
    z4.add(hub_layer6)

    # the eighth branch opreates on the third input
    # This is the fifth of the comment's highly voted comment context
    z5 = keras.Sequential()
    z5.add(hub_layer7)

    # combine the output of all the branches
    combined = keras.layers.concatenate([x.output, y.output, z.output, z1.output, z2.output, z3.output, z4.output, z5.output])

    # Build the intermediate layers
    zz = keras.layers.Dense(146, activation="relu")(combined)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(146, activation="relu")(zz)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(146, activation="relu")(zz)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(146, activation="relu")(zz)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(73, activation="relu")(zz)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(1, activation="linear")(zz)

    # our model will accept the inputs of the three branches and
    # then output a single value
    model = keras.Model(inputs=[x.input, y.input, z.input, z1.input, z2.input, z3.input, z4.input, z5.input], outputs=zz)

    model.summary()

    optimizer = keras.optimizers.Adam()

    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    
    return model

def getmodelandweights():
    model = getmodel()

    model.load_weights('./checkpoints/my_checkpoint')

    return model

def getprediction(model, titles, times, subreddits, texts, collection):
    contexts = [[datetime.utcfromtimestamp(time).minute,
                    datetime.utcfromtimestamp(time).hour,
                    datetime.utcfromtimestamp(time).day,
                    datetime.utcfromtimestamp(time).month,
                    rd.convertutctoweekdayint(time),
                    subreddit] for time, subreddit in zip(times, subreddits)]

    newest_comment = collection.find().sort([('timepostedutc',-1)]).limit(1)
    one_week_before_last_comment = math.floor((datetime.utcfromtimestamp(newest_comment[0]['timepostedutc']) - timedelta(days=1,weeks=1)).replace(hour=0, minute=0, second=0).timestamp())
    highest_voted_comments_for_week = collection.find({'timepostedutc': { '$gte': one_week_before_last_comment}}).sort([('score',-1)]).limit(5)

    dataset_comment_reference_1 = tf.data.Dataset.from_tensors([highest_voted_comments_for_week[0]['text']]*len(titles))
    dataset_comment_reference_2 = tf.data.Dataset.from_tensors([highest_voted_comments_for_week[1]['text']]*len(titles))
    dataset_comment_reference_3 = tf.data.Dataset.from_tensors([highest_voted_comments_for_week[2]['text']]*len(titles))
    dataset_comment_reference_4 = tf.data.Dataset.from_tensors([highest_voted_comments_for_week[3]['text']]*len(titles))
    dataset_comment_reference_5 = tf.data.Dataset.from_tensors([highest_voted_comments_for_week[4]['text']]*len(titles))

    #Build data pipeline
    dataset_context = tf.data.Dataset.from_tensors(contexts)
    dataset_title = tf.data.Dataset.from_tensors(titles)
    dataset_text = tf.data.Dataset.from_tensors(texts)
    dataset_inputs = tf.data.Dataset.zip((dataset_context, dataset_title, dataset_text,
        dataset_comment_reference_1, dataset_comment_reference_2, dataset_comment_reference_3,
        dataset_comment_reference_4, dataset_comment_reference_5))
    dataset = tf.data.Dataset.zip((dataset_inputs,))

    return rd.removedecimals(rd.flatten(model.predict(dataset)))
