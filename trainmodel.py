import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import random
import math
import redditdata as rd
import redditmodel as rm

settings_file = 'settings/model.json'
comments = []

for subreddit_name in rd.subreddit_list:
    comments_file = 'data/comments_' + subreddit_name + '.json'
    with open(comments_file, 'r') as f:
        comments = comments + json.load(f)

print("Highest score: {}".format(max([c['score'] for c in comments])))
print("Lowest score: {}".format(min([c['score'] for c in comments])))
print("Number of comments: {}".format(len(comments)))

random.shuffle([comments])

comment_text = [c['text'] for c in comments]
comment_scores = [c['score'] for c in comments]
max_score = max(comment_scores)
min_score = min(comment_scores)
comment_scores = rd.normalize(comment_scores, min_score, max_score)

rd.savesettings(settings_file, min_score, max_score)

comment_text_train = comment_text[math.floor(len(comment_text)*0.6):]
comment_text_test = comment_text[:math.ceil(len(comment_text)*0.4)]

comment_score_train = comment_scores[math.floor(len(comment_scores)*0.6):]
comment_score_test = comment_scores[:math.ceil(len(comment_scores)*0.4)]

model = rm.getmodel()

history = model.fit([comment_text_train], [comment_score_train], epochs=200)

results = model.evaluate([comment_text_test], [comment_score_test], verbose=0)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

model.save_weights('./checkpoints/my_checkpoint')

print("Model saved...")