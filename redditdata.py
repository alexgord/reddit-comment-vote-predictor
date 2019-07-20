import json
import math

subreddit_list = ['todayilearned', 'worldnews' ,'science', 'pics', 'gaming', 'IAmA', 'videos']

def convertsubreddittoint(subreddit):
     return subreddit_list.index(subreddit) + 1

def flatten(arr):
    return [item for sublist in arr for item in sublist]

def removedecimals(arr):
    return [math.floor(n) for n in arr]

def normalize(arr, min_val, max_val):
    return [(n - min_val)/(max_val - min_val) for n in arr]
