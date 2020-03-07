import json
import praw
from praw.models import MoreComments
from datetime import datetime
import redditdata as rd

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
        with open('data/comments_' + subreddit_name + '.json', 'w') as f:
            json.dump(comments, f)

    total_comment_num += len(comments)
    print("There are {} posts".format(numposts))
    print("There are {} comments".format(len(comments)))

print("There are {} comments in total".format(total_comment_num))
