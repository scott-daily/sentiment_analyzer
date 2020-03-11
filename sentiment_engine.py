import transformers
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
nlp_stars = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

score_dict = nlp_stars("I loved that pizza.")

print(score_dict[0])