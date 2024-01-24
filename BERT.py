import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def tokenize_and_encode(question, context):
    input_text = f"প্রশ্ন: {question} প্রস্তুতি: {context}"
    inputs = tokenizer.encode_plus(input_text, add_special_tokens=True, return_tensors="pt", max_length=max_length, truncation=True)
    return inputs

data = pd.read_csv('/mnt/f/Huggingface/rubayet/alldata.xlsx.csv')  
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

model_name = 'sagorsarker/bangla-bert-base'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

max_length = 40

def prepare_dataset(data):
    le = LabelEncoder()
    labels_start = torch.tensor(le.fit_transform(data['ANSWER'])).unsqueeze(1)  

    tokenized_data = [tokenize_and_encode(q, c) for q, c in zip(data['QUESTION'], data['sentence segment 4'])]
    max_len = max(len(item['input_ids'][0]) for item in tokenized_data)
    
    input_ids = torch.stack([torch.cat([item['input_ids'][0].long(), torch.zeros(max_len - len(item['input_ids'][0]), dtype=torch.long)], dim=0) for item in tokenized_data], dim=0)
    attention_masks = torch.stack([torch.cat([item['attention_mask'][0], torch.zeros(max_len - len(item['attention_mask'][0]))], dim=0) for item in tokenized_data], dim=0)

    dataset = TensorDataset(input_ids, attention_masks, labels_start)
    return dataset, le

train_dataset, label_encoder = prepare_dataset(train_data)
test_dataset, _ = prepare_dataset(test_data)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_dataloader) * 3  

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(3):  
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'start_positions': batch[2]}
        outputs = model(**inputs)
        if outputs.loss is not None:
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            print("Loss is None for this batch. Skipping...")

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Average Loss: {average_loss}")

model.eval()
predictions = []

for batch in tqdm(test_dataloader, desc="Evaluating"):
    batch = tuple(t.to(device) for t in batch)
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
    outputs = model(**inputs)
    start_logits = outputs.start_logits.detach().cpu().numpy()
    predictions.extend([start.argmax() for start in start_logits])


predicted_answers = label_encoder.inverse_transform(predictions)
predicted_answers = test_data['ANSWER'].dtype.type(predicted_answers)

# Evaluation metrics
print(test_data['ANSWER'],test_data['ANSWER'].type(),predicted_answers, predicted_answers.type())
accuracy = accuracy_score(test_data['ANSWER'], predicted_answers) * 100
f1 = f1_score(test_data['ANSWER'], predicted_answers, average='weighted') * 100

references = [[str(answer)] for answer in test_data['ANSWER']]
bleu_scores = [sentence_bleu(references, predicted.split()) for predicted in predicted_answers]

print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"BLEU Score: {sum(bleu_scores) / len(bleu_scores):.4f}")

