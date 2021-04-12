
import praw
import joblib
import sklearn
from praw.models import MoreComments

reddit = praw.Reddit('sentimentBot')
review_clf = joblib.load('review_clf.joblib')

all = reddit.subreddit("all")

searchTerm = 'pizza' # Get search term from form submit POST

positive_count = 0
negative_count = 0

all = reddit.subreddit("all")

for submission in all.search(searchTerm, limit=50):
    data_list = []
    data_list.append(submission.title)
    print("Post title: " + submission.title)
    prediction = review_clf.predict(data_list)
    output = prediction[0]
    if (output.item() == 1):
        positive_count += 1
    else:
        negative_count += 1

print("Positive Count: " + str(positive_count))
print("Negative Count: " + str(negative_count))