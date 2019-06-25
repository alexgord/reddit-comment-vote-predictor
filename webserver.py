from flask import Flask, request, jsonify, json, send_from_directory
import logging
import redditmodel as rm

min_value, max_value, model = rm.getmodelandweights()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_tasks():
    return send_from_directory('html', 'main.html')

@app.route('/api/predict', methods=['POST'])
def post_tasks():
    content = request.json
    texts = [content['text']]
    predictions = rm.getprediction(model, texts, min_value, max_value)
    answer = {"prediction": predictions[0]}
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
