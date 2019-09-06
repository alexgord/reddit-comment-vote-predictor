import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import numpy
from numpy import array
from datetime import datetime
import random
import math
import redditdata as rd
import redditmodel as rm

tf.enable_eager_execution()

#Build the model
model = rm.getmodel()

trainpercentage = 0.8
validationpercentage = 1 - trainpercentage

EPOCHS = 250

#Read in all the comments from disk
for subreddit_name in rd.subreddit_list:
        comments = []
        print("Training on data from {}".format(subreddit_name))
        comments_file = 'data/comments_' + subreddit_name + '.json'
        with open(comments_file, 'r') as f:
                comments = json.load(f)

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

        #Split the data into train and test sets
        comment_text_train = comment_texts[:math.floor(len(comment_texts)*trainpercentage)]
        comment_text_test = comment_texts[math.ceil(len(comment_texts)*validationpercentage):]

        comment_post_title_train = comment_post_titles[:math.floor(len(comment_post_titles)*trainpercentage)]
        comment_post_title_test = comment_post_titles[math.ceil(len(comment_post_titles)*validationpercentage):]

        comment_context_train = comment_contexts[:math.floor(len(comment_contexts)*trainpercentage)]
        comment_context_test = comment_contexts[math.ceil(len(comment_contexts)*validationpercentage):]

        comment_score_train = comment_scores[:math.floor(len(comment_scores)*trainpercentage)]
        comment_score_test = comment_scores[math.ceil(len(comment_scores)*validationpercentage):]

        #Feed the data through the model
        history = model.fit([comment_context_train, comment_post_title_train, comment_text_train],
                                [comment_score_train], epochs=EPOCHS, validation_data=([comment_context_test, comment_post_title_test, comment_text_test], [comment_score_test]))

        #Test the model and print results
        results = model.evaluate([comment_context_test, comment_post_title_test, comment_text_test], [comment_score_test], verbose=0)
        for name, value in zip(model.metrics_names, results):
                print("%s: %.3f" % (name, value))

model.save_weights('./checkpoints/my_checkpoint')

print("Model saved...")
