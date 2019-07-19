from datetime import datetime
import redditdata as rd
import redditmodel as rm
import json

model = rm.getmodelandweights()

comments = []
for subreddit_name in rd.subreddit_list:
    comments_file = 'data/comments_' + subreddit_name + '.json'
    with open(comments_file, 'r') as f:
        newcomments = json.load(f)
        comments = comments + newcomments

titles = [c['submission_title'] for c in comments]
times = [c['timepostedutc'] for c in comments]
subreddits = [rd.convertsubreddittoint(c['subreddit']) for c in comments]
texts = [c['text'] for c in comments]

predictions = rm.getprediction(model, titles, times, subreddits, texts)

predictions.sort()
predictions = rd.removedecimals(predictions)

print(predictions)

print("Max prediction: " + str(max(predictions)))
print("Min prediction: " + str(min(predictions)))
