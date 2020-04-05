from datetime import datetime
import redditdata as rd
import redditmodel as rm
import json
from pymongo import MongoClient
import sys

#mongodb setup
connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]
client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]
collection = db.comments

model = rm.getmodelandweights()

comments = [] 
database_comments = collection.find().limit(200)

for comment in database_comments:
    comments += [comment]

titles = [c['submission_title'] for c in comments]
times = [c['timepostedutc'] for c in comments]
subreddits = [rd.convertsubreddittoint(c['subreddit']) for c in comments]
texts = [c['text'] for c in comments]

predictions = rm.getprediction(model, titles, times, subreddits, texts, collection)

predictions.sort()
predictions = rd.removedecimals(predictions)

print(predictions)

print("Max prediction: " + str(max(predictions)))
print("Min prediction: " + str(min(predictions)))
