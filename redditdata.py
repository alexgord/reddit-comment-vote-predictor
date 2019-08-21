import json
import math
import operator
import praw
from datetime import datetime

MINUTES_BETWEEN_POSTS = 30
SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
HOURS_IN_DAY = 24
MINUTES_IN_DAY = MINUTES_IN_HOUR * HOURS_IN_DAY
SECONDS_IN_DAY = SECONDS_IN_MINUTE * MINUTES_IN_HOUR * HOURS_IN_DAY
NUMBER_OF_POINTS = int(SECONDS_IN_DAY / (SECONDS_IN_MINUTE * MINUTES_BETWEEN_POSTS))

subreddit_list = ['todayilearned', 'worldnews' ,'science', 'pics', 'gaming', 'IAmA', 'videos']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def convertsubreddittoint(subreddit):
     return subreddit_list.index(subreddit) + 1

def convertweekdaytostring(weekday):
     return weekdays.index(weekday) + 1

def convertutctoweekdayint(utctime):
     return convertweekdaytostring(datetime.fromtimestamp(utctime).strftime("%A"))

def flatten(arr):
    return [item for sublist in arr for item in sublist]

def removedecimals(arr):
    return [math.floor(n) for n in arr]

def normalize(arr, min_val, max_val):
    return [(n - min_val)/(max_val - min_val) for n in arr]

def dailydata(title, time, subreddit, text):
     titles = [title] * NUMBER_OF_POINTS

     minutes = list(range(0, SECONDS_IN_DAY, SECONDS_IN_MINUTE * MINUTES_BETWEEN_POSTS))
     initialtimes = [time] * NUMBER_OF_POINTS
     times = list(map(operator.add, minutes, initialtimes))

     subreddits = [subreddit] * NUMBER_OF_POINTS
     texts = [text] * NUMBER_OF_POINTS

     return titles, times, subreddits, texts

def extractInfoFromComment(comment, submission, subreddit):
     return    {
         'id' : comment.id,
         'text' : comment.body,
         'score': comment.score,
         'timepostedutc': comment.created_utc,
         'submission_title' : submission.title,
         'subreddit' : subreddit
     }
