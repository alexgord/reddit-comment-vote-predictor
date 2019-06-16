import json
import math

subreddit_list = ['todayilearned', 'worldnews' ,'science', 'pics', 'gaming', 'IAmA', 'videos']

def compartmentalize(arr):
    return [[e] for e in arr]

def flatten(arr):
    return [item for sublist in arr for item in sublist]

def removedecimals(arr):
    return [math.floor(n) for n in arr]

def denormalize(arr, min_val, max_val):
    return [n * (max_val-min_val) + min_val for n in arr]

def normalize(arr, min_val, max_val):
    return [(n - min_val)/(max_val - min_val) for n in arr]

def savesettings(file, min_val, max_val):
    settings = {'min': min_val, 'max': max_val}
    with open(file, 'w') as f:
        json.dump(settings, f)
