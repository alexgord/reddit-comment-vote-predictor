# Reddit Comment Vote Predictor

This is a small, simple tool written in Python that uses a neural network to attempt 
to learn to predict the votes for top-level comments for the more popular subreddits. 
I have chosen to train the neural network on comments from the following subreddits: 
todayilearned, worldnews, science, pics, gaming, IAmA, videos.

## Requirements

This tool was developed using TensorFlow 2.0

## Usage

### Scripts and functions

To scrape data from Reddit to get your own data to train the neural network with, 
run the `queryreddit.py` script.

To create, train, and save the neural network using this data, run the `trainmodel.py` 
script.

To test the neural network and predict the votes on the downloaded comments, run 
the `predict.py` script.

The code to load the model once it is created and saved is the following: 

```python
import redditmodel as rm

model = rm.getmodelandweights()
```

### REST API

There is a REST API available. The URL to access it is `http://www.votepredictor.com/api/predict`. 
The API expects a POST request with JSON data containing the following fields:

* time: The unix time in seconds since the Unix epoch
* title: The title of the post you are commenting on
* text: The comment itself
* subreddit: An integer between 1 and 7 representing which of the subreddits the comment will be 
posted to. In order from 1 to 7, the subreddits are: todayilearned, worldnews, science, pics, gaming, IAmA, videos.

The API will then respond with a JSON object containing a `prediction` field, which will have an 
integer representing the votes the neural network predicts this comment will have.

Here is an example CURL command that requests a prediction: 

```bash
curl -X POST http://www.votepredictor.com/api/predict --data "{\"time\": 1563314096, \"title\": \"some title\", \"text\": \"some text\", \"subreddit\": 1}" --header "Content-Type: application/json"
```

Similarly, `http://www.votepredictor.com/api/predict` can be used in the exact same way to return 
the predictions over the next 24 hours.

## Future work

* I plan to have this tool automatically update the neural network with new comments on 
reddit so it can continue to learn.
* I plan to create a chrome plugin that automatically informs users of the tool's 
prediction when they write a top-level comment on Reddit.
* I plan to create a second neural network to complement the first I created. This 
neural network will generate text based on the most highly voted Reddit comments.
