import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import praw

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
nlp_stars_sentiment = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

'''score_dict = nlp_stars_sentiment("Smaller jar, no peanut butter all over my fingers and wrists. Fresher. Was buying larger jars to get better pricing, but so messy trying to get it out near middle to end. Plus can be weeks before I eat again. This is less messy and still cost effective. I prefer Skippy, but I like this as 2nd choice.")'''

'''print(score_dict[0])'''

reddit = praw.Reddit('sentimentBot')


for submission in reddit.subreddit('coronavirus').hot(limit=10):
    submission_dict = nlp_stars_sentiment(submission.title)
    print(submission.title)
    print(submission_dict[0])