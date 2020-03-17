from pymongo import MongoClient
import json
import redditdata as rd
import sys

connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]

client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]

collection = db.comments

collection.create_index("comment_id")

print("Gathering comments...")
comments = []
removed_comments_file = "data/science_removed_comments_data.json"
with open(removed_comments_file, 'r') as f:
    comments = json.load(f)

print("Writing comments to database...")
for comment in comments:
    result = collection.count_documents({'comment_id':{'$eq':comment['comment_id']}})
    if result == 0:
        collection.insert_one(comment)
    else:
        collection.update_one({'comment_id':comment['comment_id']}, {"$set": {"removed": comment['removed']}}, upsert=True)

print("Comments written to database")
