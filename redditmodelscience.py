import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import tensorflow_hub as hub
import redditdata as rd
import json

checkpoint_dir = './science_training_checkpoints/my_checkpoint'

def getmodel():
    dropoutrate = 0.2

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
    zz = keras.layers.Dense(40, activation="relu")(combined)
    zz = keras.layers.Dropout(dropoutrate)(zz)
    zz = keras.layers.Dense(30, activation="relu")(zz)
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

def getprediction(model, titles, times, subreddits, texts):
    contexts = [[datetime.utcfromtimestamp(time).minute,
                datetime.utcfromtimestamp(time).hour,
                datetime.utcfromtimestamp(time).day,
                datetime.utcfromtimestamp(time).month,
                rd.convertutctoweekdayint(time),
                subreddit] for time, subreddit in zip(times, subreddits)]

    return rd.removedecimals(rd.flatten(model.predict([contexts, titles, texts])))
