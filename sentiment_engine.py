import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import praw

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
nlp_stars_sentiment = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

'''score_dict = nlp_stars_sentiment("Smaller jar, no peanut butter all over my fingers and wrists. Fresher. Was buying larger jars to get better pricing, but so messy trying to get it out near middle to end. Plus can be weeks before I eat again. This is less messy and still cost effective. I prefer Skippy, but I like this as 2nd choice.")'''

'''print(score_dict[0])'''

reddit = praw.Reddit('sentimentBot')

all = reddit.subreddit("all")
amazon_reviews = reddit.subreddit("amazonreviews")

for submission in amazon_reviews.new(limit=3):
    print("title:" + submission.title)
    submission.comments.replace_more()
    for comment in submission.comments.list():
        print("Comment Body: " + comment.body)
        print(nlp_stars_sentiment(comment.body))

print(nlp_stars_sentiment("The tire shop was great.  The service was very helpful.  I did not like that I had to wait a little long, but overall it was good."))


'''for submission in reddit.subreddit('all').hot(limit=10):
    submission_dict = nlp_stars_sentiment(submission.title)
    print(submission.title)
    print(submission_dict[0])'''