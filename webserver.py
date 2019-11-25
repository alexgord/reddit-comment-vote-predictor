import time, threading
from flask import Flask, request, jsonify, json, send_from_directory
import logging
import redditmodel as rm
import redditmodelgenerative as rmg
import redditmodelscience as rms
import redditdata as rd
import tensorflow as tf
import pickle
import praw
from praw.models import MoreComments

tf.enable_eager_execution()

SECONDS_IN_MINUTE = 60
MINUTES_IN_HOUR = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * MINUTES_IN_HOUR

#Mode which predicts votes
model = rm.getmodelandweights()

#Get information to build model which generates text
f = open('settings/char2idx.pckl', 'rb')
char2idx = pickle.load(f)
f.close()

print("Read char2idx from disk")

f = open('settings/idx2char.pckl', 'rb')
idx2char = pickle.load(f)
f.close()

print("Read idx2char from disk")

f = open('settings/vocab.pckl', 'rb')
vocab = pickle.load(f)
f.close()

print("Read vocab from disk")

# Length of the vocabulary in chars
vocab_size = len(vocab)

#Model which generates text
modelgenerative = rmg.getmodel(vocab_size = vocab_size, embedding_dim=rmg.embedding_dim, rnn_units=rmg.rnn_units, batch_size=1)

modelgenerative.load_weights(rmg.checkpoint_dir)

modelgenerative.build(tf.TensorShape([1, None]))

#Model which predicts if a comment will be removed on /r/science
modelscience = rms.getmodelandweights()

commentstoremove = []

def get_comments_to_remove_timed():
    while True:
        print(time.ctime())
        get_comments_to_remove()
        time.sleep(SECONDS_IN_HOUR)

def get_comments_to_remove():
    global commentstoremove
    print("Getting comments from reddit...")
    reddit = praw.Reddit('redditaiscraper', user_agent='redditaiscraper script by thecomputerscientist')

    subreddit_name = "science"

    comments = []
    titles = []
    texts = []

    for submission in reddit.subreddit(subreddit_name).new(limit = 100):
        # Removes all MorComment objects from submission object
        submission.comments.replace_more(limit = None)

        unremoved_comments = [c for c in submission.comments.list() if c.body != "[removed]"]

        for comment in unremoved_comments:
            comments +=  [rd.extractInfoFromComment(comment, submission, subreddit_name)]
            titles += [submission.title]
            texts += [comment.body]

    print("Got all submissions...")

    predictions = rms.getprediction(modelscience, titles, texts)

    for comment, prediction in zip(comments, predictions):
        comment.update( {"removal_prediction":bool(prediction)})

    print("Found all comments to be removed")
    commentstoremove = [comment for comment in comments if comment['removal_prediction']]

app = Flask(__name__)

@app.before_first_request
def activate_job():
    print("Before first request")
    thread = threading.Thread(target=get_comments_to_remove_timed)
    thread.start()

@app.route('/', methods=['GET'])
def get_tasks():
    return send_from_directory('html', 'main.html')

@app.route('/api/subreddits', methods=['GET'])
def subreddit_task():
    return jsonify(rd.subreddit_list)

@app.route('/api/predict', methods=['POST'])
def post_predict():
    answer = {}
    content = request.json
    if 'time' in content and 'title' in content and 'text' in content and 'subreddit' in content:
        time = [content['time']]
        title = [content['title']]
        text = [content['text']]
        subreddit = [content['subreddit']]
        if subreddit[0] > 0 and subreddit[0] <= len(rd.subreddit_list):
            predictions = rm.getprediction(model, title, time, subreddit, text)
            answer = {"prediction": predictions[0]}
        else:
            answer = {"error": "Subreddit must be an integer between 1 and " + str(len(rd.subreddit_list))}
    else:
        answer = {"error": "Missing one or more fields. Please provide time, title, text, and subreddit"}
    return jsonify(answer)

@app.route('/api/predict/day', methods=['POST'])
def post_predict_day():
    answer = {}
    content = request.json
    if 'time' in content and 'title' in content and 'text' in content and 'subreddit' in content:
        time = content['time']
        title = content['title']
        text = content['text']
        subreddit = content['subreddit']
        if subreddit > 0 and subreddit <= len(rd.subreddit_list):
            titles, times, subreddits, texts = rd.dailydata(title, time, subreddit, text)
            predictions = rm.getprediction(model, titles, times, subreddits, texts)
            answer = {"times": times, "predictions": predictions}
        else:
            answer = {"error": "Subreddit must be an integer between 1 and " + str(len(rd.subreddit_list))}
    else:
        answer = {"error": "Missing one or more fields. Please provide time, title, text, and subreddit"}
    return jsonify(answer)

@app.route('/api/generate', methods=['POST'])
def post_generate():
    answer = {}
    content = request.json
    if 'text' in content:
        text = content['text']
        generatedtext = rmg.generatesentence(modelgenerative, text, char2idx, idx2char)
        answer = {"generated_text": generatedtext}
    else:
        answer = {"error": "Missing one or more fields. Please provide time, title, text, and subreddit"}
    return jsonify(answer)

@app.route('/api/science/badcomments', methods=['POST'])
def post_predict_removed():
    return jsonify(commentstoremove)

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)

@app.route('/<path:path>')
def send_app(path):
    return send_from_directory('html', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)

if __name__ == '__main__':
    app.run(debug=True)
