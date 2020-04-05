import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import json
import numpy
from numpy import array
from datetime import datetime, timedelta
import random
import math
import redditdata as rd
import redditmodel as rm
import sys

#Build the model
model = rm.getmodel()

trainpercentage = 0.85
validationpercentage = 1 - trainpercentage

EPOCHS = 1200

connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]
client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]
collection = db.comments

#Read in all the comments from the database
number_of_documents = collection.count_documents({'comment_id': { '$exists': True}})

print("Gathering {} documents from database".format(number_of_documents))

results = collection.find()

comments = []
daily_comments = dict()
for comment in results:
    day = rd.getdayssinceepoch(comment['timepostedutc'])
    if day in daily_comments.keys():
        daily_comments[day] += [comment]
    else:
        daily_comments[day] = [comment]
    #highest_voted_comments_for_week = collection.find({'$and':[{'timepostedutc': {'$lt': comment['timepostedutc']}},{'timepostedutc': {'$gte': one_week_ago}}]}).sort([('score',-1)]).limit(5)
    #comment['highest_voted_comments_for_week'] = [comment['text'] for comment in highest_voted_comments_for_week]
    comments += [comment]

relevant_context_comments_for_day = {}

for day in daily_comments:
    if day not in relevant_context_comments_for_day.keys():
        relevant_context_comments_for_day[day] = []
    for d in range(-7,1):
        pastday = day - d
        if pastday in daily_comments.keys():
            relevant_context_comments_for_day[day] += daily_comments[pastday]
    relevant_context_comments_for_day[day] = sorted(relevant_context_comments_for_day[day], key = lambda i: i['score'], reverse=True)[:5]

for comment in comments:
    highest_voted_comments_for_week = relevant_context_comments_for_day[rd.getdayssinceepoch(comment['timepostedutc'])]
    highest_voted_comments_text = [c['text'] for c in highest_voted_comments_for_week]
    if len(highest_voted_comments_text) < 5:
        l = len(highest_voted_comments_text)
        for i in range(0,5-l):
            highest_voted_comments_text += [""]
    comment['highest_voted_comments_for_week'] = highest_voted_comments_text

print("{} comments in memory".format(len(comments)))

#Print a few statistics on the data
print("Highest score: {}".format(max([c['score'] for c in comments])))
print("Lowest score: {}".format(min([c['score'] for c in comments])))
print("Number of comments: {}".format(len(comments)))

#Prepare data to be fed into the model
random.shuffle(comments)

#Split data into a series of lists to be fed into the model
comment_post_titles = [c['submission_title'] for c in comments]
comment_contexts = [[datetime.utcfromtimestamp(c['timepostedutc']).minute,
                        datetime.utcfromtimestamp(c['timepostedutc']).hour,
                        datetime.utcfromtimestamp(c['timepostedutc']).day,
                        datetime.utcfromtimestamp(c['timepostedutc']).month,
                        rd.convertutctoweekdayint(c['timepostedutc']),
                        rd.convertsubreddittoint(c['subreddit'])] for c in comments]
comment_texts = [c['text'] for c in comments]
comment_scores = [c['score'] for c in comments]
comment_highly_voted_reference_1 = [c['highest_voted_comments_for_week'][0] for c in comments]
comment_highly_voted_reference_2 = [c['highest_voted_comments_for_week'][1] for c in comments]
comment_highly_voted_reference_3 = [c['highest_voted_comments_for_week'][2] for c in comments]
comment_highly_voted_reference_4 = [c['highest_voted_comments_for_week'][3] for c in comments]
comment_highly_voted_reference_5 = [c['highest_voted_comments_for_week'][4] for c in comments]

#Split the data into train and test sets
comment_text_train = comment_texts[:math.floor(len(comment_texts)*trainpercentage)]
comment_text_test = comment_texts[math.ceil(len(comment_texts)*validationpercentage):]

comment_post_title_train = comment_post_titles[:math.floor(len(comment_post_titles)*trainpercentage)]
comment_post_title_test = comment_post_titles[math.ceil(len(comment_post_titles)*validationpercentage):]

comment_context_train = comment_contexts[:math.floor(len(comment_contexts)*trainpercentage)]
comment_context_test = comment_contexts[math.ceil(len(comment_contexts)*validationpercentage):]

comment_score_train = comment_scores[:math.floor(len(comment_scores)*trainpercentage)]
comment_score_test = comment_scores[math.ceil(len(comment_scores)*validationpercentage):]

comment_highly_voted_reference_1_train = comment_highly_voted_reference_1[:math.floor(len(comment_highly_voted_reference_1)*trainpercentage)]
comment_highly_voted_reference_1_test = comment_highly_voted_reference_1[math.ceil(len(comment_highly_voted_reference_1)*validationpercentage):]

comment_highly_voted_reference_2_train = comment_highly_voted_reference_2[:math.floor(len(comment_highly_voted_reference_2)*trainpercentage)]
comment_highly_voted_reference_2_test = comment_highly_voted_reference_2[math.ceil(len(comment_highly_voted_reference_2)*validationpercentage):]

comment_highly_voted_reference_3_train = comment_highly_voted_reference_3[:math.floor(len(comment_highly_voted_reference_3)*trainpercentage)]
comment_highly_voted_reference_3_test = comment_highly_voted_reference_3[math.ceil(len(comment_highly_voted_reference_3)*validationpercentage):]

comment_highly_voted_reference_4_train = comment_highly_voted_reference_4[:math.floor(len(comment_highly_voted_reference_4)*trainpercentage)]
comment_highly_voted_reference_4_test = comment_highly_voted_reference_4[math.ceil(len(comment_highly_voted_reference_4)*validationpercentage):]

comment_highly_voted_reference_5_train = comment_highly_voted_reference_5[:math.floor(len(comment_highly_voted_reference_5)*trainpercentage)]
comment_highly_voted_reference_5_test = comment_highly_voted_reference_5[math.ceil(len(comment_highly_voted_reference_5)*validationpercentage):]

#Build training data pipeline
dataset_context_train = tf.data.Dataset.from_tensors(comment_context_train)
dataset_title_train = tf.data.Dataset.from_tensors(comment_post_title_train)
dataset_text_train = tf.data.Dataset.from_tensors(comment_text_train)
dataset_score_train = tf.data.Dataset.from_tensors(comment_score_train)
dataset_comment_reference_1_train = tf.data.Dataset.from_tensors(comment_highly_voted_reference_1_train)
dataset_comment_reference_2_train = tf.data.Dataset.from_tensors(comment_highly_voted_reference_2_train)
dataset_comment_reference_3_train = tf.data.Dataset.from_tensors(comment_highly_voted_reference_3_train)
dataset_comment_reference_4_train = tf.data.Dataset.from_tensors(comment_highly_voted_reference_4_train)
dataset_comment_reference_5_train = tf.data.Dataset.from_tensors(comment_highly_voted_reference_5_train)
dataset_inputs_train = tf.data.Dataset.zip((dataset_context_train, dataset_title_train, dataset_text_train,
dataset_comment_reference_1_train, dataset_comment_reference_2_train,
dataset_comment_reference_3_train, dataset_comment_reference_4_train, dataset_comment_reference_5_train))
dataset_train = tf.data.Dataset.zip((dataset_inputs_train, dataset_score_train))

#Build testing data pipeline
dataset_context_test = tf.data.Dataset.from_tensors(comment_context_test)
dataset_title_test = tf.data.Dataset.from_tensors(comment_post_title_test)
dataset_text_test = tf.data.Dataset.from_tensors(comment_text_test)
dataset_score_test = tf.data.Dataset.from_tensors(comment_score_test)
dataset_comment_reference_1_test = tf.data.Dataset.from_tensors(comment_highly_voted_reference_1_test)
dataset_comment_reference_2_test = tf.data.Dataset.from_tensors(comment_highly_voted_reference_2_test)
dataset_comment_reference_3_test = tf.data.Dataset.from_tensors(comment_highly_voted_reference_3_test)
dataset_comment_reference_4_test = tf.data.Dataset.from_tensors(comment_highly_voted_reference_4_test)
dataset_comment_reference_5_test = tf.data.Dataset.from_tensors(comment_highly_voted_reference_5_test)
dataset_inputs_test= tf.data.Dataset.zip((dataset_context_test, dataset_title_test, dataset_text_test,
dataset_comment_reference_1_test, dataset_comment_reference_2_test, dataset_comment_reference_3_test,
dataset_comment_reference_4_test, dataset_comment_reference_5_test))
dataset_test = tf.data.Dataset.zip((dataset_inputs_test, dataset_score_test))

#Feed the data through the model
history = model.fit(dataset_train, epochs=EPOCHS, validation_data=dataset_test)

#Test the model and print results
print("Model evaluation:")
results = model.evaluate(dataset_test, verbose=0)
for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))

model.save_weights('./checkpoints/my_checkpoint')

print("Model saved...")
