import json

comments_file = 'data/comments.json'

with open(comments_file, 'r') as f:
    comments = json.load(f)

scores = [e['score'] for e in comments]

greater_than_100 = sum(e > 100 for e in scores)
less_than_100 = sum(e < 100 for e in scores)

print("Number of comments with score > 100 {}".format(greater_than_100))
print("Number of comments with score < 100 {}".format(less_than_100))