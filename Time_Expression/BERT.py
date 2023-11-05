  # BEGIN: 4j3d9f8hj3d9
import torch
from transformers import BertForTokenClassification, BertTokenizer

  # Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# Set the number of labels for the classification task
num_labels = 3

# Modify the pre-trained BERT model to output the correct number of labels
model.classifier = torch.nn.Linear(model.classifier.in_features, num_labels)

# Define the optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example usage:
text = "Hello, my name is John."
tokens = tokenizer.tokenize(text)
inputs = tokenizer.encode(text, return_tensors="pt").to(device)
outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)
label_list = ['PERSON', 'LOCATION', 'ORGANIZATION']
print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])

