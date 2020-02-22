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

MAX_LEVEL = 3

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
    numposts += 1
    try:
        for top_level_comment in submission.comments:
            print("Getting top level comment")
            comments += retrieveComments(top_level_comment, submission, subreddit_name, 0, MAX_LEVEL)
    except (praw.exceptions.APIException, PrawcoreException):
        print("Had problems getting comments for post, moving on to next post...")

if len(comments) > 0:
    with open('data/' + subreddit_name + '_removed_comments_data.json', 'w') as f:
        json.dump(comments, f)

print("Comments written to disk")

print("There are {} posts".format(numposts))
print("There are {} comments".format(len(comments)))
