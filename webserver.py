from flask import Flask, request, jsonify, json, send_from_directory
import logging
import redditmodel as rm
import redditmodelgenerative as rmg
import redditdata as rd
import tensorflow as tf
import pickle

tf.enable_eager_execution()

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

modelgenerative.load_weights(tf.train.latest_checkpoint(rmg.checkpoint_dir))

modelgenerative.build(tf.TensorShape([1, None]))

app = Flask(__name__)

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
