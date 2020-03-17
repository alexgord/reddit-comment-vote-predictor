import tensorflow as tf
from tensorflow import keras
import json
from pymongo import MongoClient
import sys
import redditdata as rd
import redditmodelscience as rms
import random
import math
from os import listdir
from os.path import isfile, join

removed_comments_file = "data/science_removed_comments_data.json"

comments = []

trainpercentage = 0.85
validationpercentage = 1 - trainpercentage

epochs = 2000

#Build the model
model = rms.getmodel()

connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]
client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]
collection = db.comments

#Read in all the comments from the database
number_of_documents = collection.count_documents({'$and' : [{'removed': { '$exists': True}}, {'subreddit': {'$eq': 'science'}}]})

print("Gathering {} documents from database".format(number_of_documents))

results = collection.find({'$and' : [{'removed': { '$exists': True}}, {'subreddit': {'$eq': 'science'}}]})

comments = []
for comment in results:
    comments += [comment]

print("{} comments in memory".format(len(comments)))

random.shuffle(comments)

comment_titles = [c['submission_title'] for c in comments]
comment_texts = [c['text'] for c in comments]
comment_removed = [c['removed'] for c in comments]

#Split the data into train and test sets
comment_post_title_train = comment_titles[:math.floor(len(comment_titles)*trainpercentage)]
comment_post_title_test = comment_titles[math.ceil(len(comment_titles)*validationpercentage):]

comment_text_train = comment_texts[:math.floor(len(comment_texts)*trainpercentage)]
comment_text_test = comment_texts[math.ceil(len(comment_texts)*validationpercentage):]

comment_removed_train = comment_removed[:math.floor(len(comment_removed)*trainpercentage)]
comment_removed_test = comment_removed[math.ceil(len(comment_removed)*validationpercentage):]

#Build training data pipeline
dataset_title_train = tf.data.Dataset.from_tensors(comment_post_title_train)
dataset_text_train = tf.data.Dataset.from_tensors(comment_text_train)
dataset_removed_train = tf.data.Dataset.from_tensors(comment_removed_train)
dataset_inputs_train = tf.data.Dataset.zip((dataset_title_train, dataset_text_train))
dataset_train = tf.data.Dataset.zip((dataset_inputs_train, dataset_removed_train))

#Build testing data pipeline
dataset_title_test = tf.data.Dataset.from_tensors(comment_post_title_test)
dataset_text_test = tf.data.Dataset.from_tensors(comment_text_test)
dataset_removed_test = tf.data.Dataset.from_tensors(comment_removed_test)
dataset_inputs_test = tf.data.Dataset.zip((dataset_title_test, dataset_text_test))
dataset_test = tf.data.Dataset.zip((dataset_inputs_test, dataset_removed_test))

#Feed the data through the model
history = model.fit(dataset_train, epochs=epochs, validation_data = dataset_test)

#Test the model and print results
print("Model evaluation:")
results = model.evaluate(dataset_test, verbose=0)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

model.save_weights(rms.checkpoint_dir)

print("Model saved...")