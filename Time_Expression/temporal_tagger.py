
import torch
from transformers import AutoTokenizer, BertForTokenClassification

input_text = [
  "I will meet you tomorrow at 3pm.",
  "The event happened on 12th of June, 2021.",
  "I have been working here for 2 years.",
  "The movie was released in 1995.",
  "I will be on vacation from July 1st to July 10th."
]


# Define the tag to label mapping
tag2label = [
  "O",
  "B-TIME",
  "I-TIME",
  "B-DATE",
  "I-DATE",
  "B-DURATION",
  "I-DURATION",
  "B-SET",
  "I-SET"
]

tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier",padding=True, truncation=True , use_fast=False)
model = BertForTokenClassification.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier")
processed_text = tokenizer(input_text, padding=True, truncation=True ,return_tensors="pt")
result = model(**processed_text)

# Convert the model's output to the desired format
output = []
for i, sentence in enumerate(input_text):
  tokens = tokenizer.tokenize(sentence)
  predictions = torch.argmax(result.logits[i], dim=1)
  tags = [tag2label[prediction] for prediction in predictions]
  sentence_output = []
  for j, token in enumerate(tokens):
    sentence_output.append((token, tags[j]))
  output.append(sentence_output)

# Print the output
for sentence in output:
  print([(token, tag) for token, tag in sentence])

print(result)