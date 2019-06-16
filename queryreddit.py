import json
import praw
from praw.models import MoreComments
from datetime import datetime
import redditdata as rd

reddit = praw.Reddit('redditaiscraper', user_agent='redditaiscraper script by thecomputerscientist')

print("Reddit AI scraper\n\n")
print("Scraping commencing...")

total_comment_num = 0

for subreddit_name in rd.subreddit_list:
    print("Gettings posts from: {}".format(subreddit_name))
    numposts = 0
    numcomments = 0
    comments = []
    # assume you have a Reddit instance bound to variable `reddit`
    subreddit = reddit.subreddit(subreddit_name)

    posts = subreddit.hot()

    for submission in posts:
        numposts += 1
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments) or top_level_comment.body == "[removed]":
                continue
            numcomments += 1
            comments += [{
                            'id' : top_level_comment.id,
                            'text' : top_level_comment.body,
                            'score': top_level_comment.score,
                            'timepostedutc': top_level_comment.created_utc,
                            'submission_title' : submission.title
                        }]

    with open('data/comments_' + subreddit_name + '.json', 'w') as f:
        json.dump(comments, f)

    total_comment_num += numcomments
    print("There are {} posts".format(numposts))
    print("There are {} comments".format(numcomments))

print("There are {} comments in total".format(total_comment_num))
