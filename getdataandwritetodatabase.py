from pymongo import MongoClient
import json
import praw
from praw.models import MoreComments
from datetime import datetime
import redditdata as rd
import sys

def retrieveComments(comment, submission, subreddit_name, level, max_level):
    if isinstance(comment, MoreComments) or comment.score_hidden or comment.body == "[removed]" or level > max_level:
        return []

    returnedcomments = [rd.extractInfoFromComment(comment, submission, subreddit_name)]

    for reply in comment.replies:
        returnedcomments += retrieveComments(reply, submission, subreddit_name, level + 1, max_level)

    return returnedcomments

appId = ''
appSecret = ''

app_settings_file = 'settings/app.json'
with open(app_settings_file, 'r') as f:
    settings = json.load(f)
    appId = settings['app_id']
    appSecret = settings['secret']

reddit = praw.Reddit(client_id=appId, client_secret=appSecret, user_agent=rd.user_agent)

print("Reddit AI scraper\n\n")
print("Scraping commencing...")

total_comment_num = 0

MAX_LEVEL = 6
POST_LIMIT = 1000

connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]

client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]

collection = db.comments

collection.create_index("comment_id")

for subreddit_name in rd.subreddit_list:
    print("Gettings posts from: {}".format(subreddit_name))
    numposts = 0
    comments = []
    subreddit = reddit.subreddit(subreddit_name)

    posts = subreddit.hot(limit=POST_LIMIT)

    for submission in posts:
        submission.comments.replace_more(limit=0)
        numposts += 1
        for top_level_comment in submission.comments:
            comments += retrieveComments(top_level_comment, submission, subreddit_name, 0, MAX_LEVEL)

if len(comments) > 0:
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
