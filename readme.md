# Reddit Comment Vote Predictor

This is a small, simple tool written in Python that uses a neural network to attempt 
to learn to predict the votes for top-level comments for the more popular subreddits. 
I have chosen to train the neural network on comments from the following subreddits: 
todayilearned, worldnews ,science, pics, gaming, IAmA, videos.

## Requirements

This tool was developed using TensorFlow 2.0

## Usage

To scrape data from Reddit to get your own data to train the neural network with, 
run the `queryreddit.py` script.

To create, train, and save the neural network using the data, run the `trainmodel.py` 
script.

To test the neural network and predict the votes on some sample dummy comments, run 
the `predict.py` script.

The code to load the model once it is created and saved is the following: 

```python
import redditdata as rd
import redditmodel as rm
import json

settings_file = 'settings/model.json'

with open(settings_file, 'r') as f:
    settings = json.load(f)

model = rm.getmodel()

model.load_weights('./checkpoints/my_checkpoint')
```

## Future work

* I plan to have this tool automatically update the neural network with new comments on 
reddit so it can continue to learn.
* I plan to create a web portal that allows users to paste in text and receive a 
prediction of the number of votes such a comment would receive.
* I plan to create a chrome plugin that automatically informs users of the tool's 
prediction when they write a top-level comment on reddit.
* I Plan to create a Python API that allows users to access local and cloud copies 
of this tool to leverage predictions in their own scripts.