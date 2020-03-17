from pymongo import MongoClient
import json
import praw
from praw.models import MoreComments
from prawcore.exceptions import PrawcoreException
from datetime import datetime
import redditdata as rd
import getpass
import sys
import time

print("Reddit AI science scraper\n\n")

appId = ''
appSecret = ''
username = ''
password = ''
tfacode = ''

app_settings_file = 'settings/app.json'
with open(app_settings_file, 'r') as f:
    settings = json.load(f)
    appId = settings['app_id']
    appSecret = settings['secret']

print("Login credentials required\n\n")

username = input("Username: ")
password = getpass.getpass()

tfacode = input("Two factor authentication code (blank if not applicable): ")

if tfacode != '':
    password += ':' + tfacode

reddit = praw.Reddit(client_id=appId, client_secret=appSecret, password=password, user_agent=rd.user_agent, username=username)

connection_string = "mongodb://localhost:27017" if len(sys.argv) == 1 else sys.argv[1]

client = MongoClient(connection_string)
db=client["reddit-comment-vote-predictor"]

collection = db.comments

collection.create_index("comment_id")

try:
    if username == '' or reddit.user.me() != username:
        print("Login failure, exiting...")
        sys.exit()
    else:
        print("Login successful.")
except:
    print("Login failure, exiting...")
    sys.exit()

print("\n\nScraping commencing...")

POSTS_LIMIT = 1000

total_comment_num = 0

subreddit_name = 'science'

MAX_LEVEL = 6

print("Gettings posts from: {}".format(subreddit_name))
numposts = 0
comments = []
subreddit = reddit.subreddit(subreddit_name)

posts = subreddit.hot(limit=POSTS_LIMIT)

def retrieveComments(comment, submission, subreddit_name, level, max_level):
    if isinstance(comment, MoreComments) or comment.score_hidden or comment.body == "[removed]" or level > max_level:
        return []

    returnedcomments = [rd.extractInfoFromCommentForScience(comment, submission, subreddit_name)]

    for reply in comment.replies:
        returnedcomments += retrieveComments(reply, submission, subreddit_name, level + 1, max_level)
    
    return returnedcomments

for submission in posts:
    print("Getting comments from post: " + submission.title)
    try:
        submission.comments.replace_more(limit=0)
        numposts += 1
        for top_level_comment in submission.comments:
            print("Getting top level comment")
            comments += retrieveComments(top_level_comment, submission, subreddit_name, 0, MAX_LEVEL)
    except (praw.exceptions.APIException, PrawcoreException):
        print("Had problems getting comments for post, moving on to next post...")

if len(comments) > 0:
    print("Writing comments to database...")
    for comment in comments:
        result = collection.count_documents({'comment_id':{'$eq':comment['comment_id']}})
        if result == 0:
            collection.insert_one(comment)
        else:
            collection.update_one({'comment_id':comment['comment_id']}, {"$set": {"removed": comment['removed']}}, upsert=True)

    print("Comments written to database")
else:
    print("No comments written to database")
