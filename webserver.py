from flask import Flask, request, jsonify, json, send_from_directory
import logging
import redditmodel as rm
import redditdata as rd

model = rm.getmodelandweights()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_tasks():
    return send_from_directory('html', 'main.html')

@app.route('/api/subreddits', methods=['GET'])
def subreddit_task():
    return jsonify(rd.subreddit_list)

@app.route('/api/predict', methods=['POST'])
def post_tasks():
    answer = {}
    content = request.json
    if 'time' in content and 'title' in content and 'text' in content and 'subreddit' in content:
        time = [content['time']]
        title = [content['title']]
        text = [content['text']]
        subreddit = [content['subreddit']]
        if subreddit[0] > 0 and subreddit[0] < len(rd.subreddit_list):
            predictions = rm.getprediction(model, title, time, subreddit, text)
            answer = {"prediction": predictions[0]}
        else:
            answer = {"error": "Subreddit must be an integer between 1 and " + str(len(rd.subreddit_list))}
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
