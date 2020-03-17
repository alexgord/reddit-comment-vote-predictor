from pymongo import MongoClient
import json
import redditdata as rd
import sys

connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]

client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]

collection = db.comments

collection.create_index("comment_id")

comments = []
#Read in all the comments from disk
for subreddit_name in rd.subreddit_list:
        print("Reading data from {}".format(subreddit_name))
        comments_file = 'data/comments_' + subreddit_name + '.json'
        with open(comments_file, 'r') as f:
                comments += json.load(f)

print("Finding comments to insert in database...")
comments_to_insert = []
for comment in comments:
    result = collection.count_documents({'comment_id':{'$eq':comment['comment_id']}})
    if result == 0:
        comments_to_insert += [comment]

if len(comments_to_insert) > 0:
    print("inserting {} comments...".format(len(comments_to_insert)))
    collection.insert_many(comments_to_insert)
    print("Comments inserted")
else:
    print("No comments to insert")
