import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import redditdata as rd
import json

def getmodel():
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1", output_shape=[20],
                           input_shape=[], dtype=tf.string, trainable=True)

    model = keras.Sequential()
    model.add(hub_layer)
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))

    model.summary()

    optimizer = keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    
    return model

def getmodelandweights():
    settings_file = 'settings/model.json'

    with open(settings_file, 'r') as f:
        settings = json.load(f)

    model = getmodel()

    model.load_weights('./checkpoints/my_checkpoint')

    return settings['min'], settings['max'], model

def getprediction(model, texts, min_value, max_value):
    texts = [texts]

    predictions = rd.flatten(model.predict(texts))

    return rd.removedecimals(rd.denormalize(predictions, min_value, max_value))
