import json
import praw
from praw.models import MoreComments
from datetime import datetime
import redditdata as rd

def extractInfoFromComment(comment, subreddit):
    return {
        'id' : comment.id,
        'text' : comment.body,
        'score': comment.score,
        'timepostedutc': comment.created_utc,
        'submission_title' : submission.title,
        'subreddit' : subreddit,
        'removed' : comment.banned_by != None,
        'mod': comment._mod,
        'approved_at_utc': comment.approved_at_utc,
        'mod_reason_by': comment.mod_reason_by,
        'banned_by': comment.banned_by,
        'removal_reason': comment.removal_reason,
        'user_reports': comment.user_reports,
        'banned_at_utc': comment.banned_at_utc,
        'mod_reason_title': comment.mod_reason_title,
        'report_reasons': comment.report_reasons,
        'approved_by': comment.approved_by,
        'num_reports': comment.num_reports,
        'locked': comment.locked,
        'collapsed': comment.collapsed,
        'mod_reports': comment.mod_reports,
        'mod_note': comment.mod_note
    }

print("Reddit AI scraper\n\n")
print("Scraping commencing...")

POSTS_LIMIT = 100
    
total_comment_num = 0

reddit = praw.Reddit('redditaiscraper', user_agent='redditaiscraper script by thecomputerscientist')

subreddit_name = 'science'

print("Gettings posts from: {}".format(subreddit_name))
numposts = 0
numcomments = 0
comments = []
subreddit = reddit.subreddit(subreddit_name)

posts = subreddit.hot(limit=POSTS_LIMIT)

for submission in posts:
    numposts += 1
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments) or top_level_comment.score_hidden:
            continue
        numcomments += 1
        comments += [extractInfoFromComment(top_level_comment, subreddit_name)]
    #if not isinstance(top_level_comment, MoreComments) and top_level_comment.body != "[removed]"  and not top_level_comment.score_hidden:
    #    for second_level_comment in top_level_comment.replies:
    #        if isinstance(second_level_comment, MoreComments) or second_level_comment.body == "[removed]"  or second_level_comment.score_hidden:
    #            continue
    #        numcomments += 1
    #        comments += [extractInfoFromComment(second_level_comment, subreddit_name)]

with open('data/comments_' + subreddit_name + '_mod_removed_data.json', 'w') as f:
    json.dump(comments, f)

total_comment_num += numcomments
print("There are {} posts".format(numposts))
print("There are {} comments".format(numcomments))

print("There are {} comments in total".format(total_comment_num))
