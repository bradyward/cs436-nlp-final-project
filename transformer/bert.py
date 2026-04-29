import os
import re
import numpy as np 
from sklearn.metrics import accuracy_score

import transformers
from transformers import BertTokenizer, BertModel

import torch
from torch import cuda
from tqdm.notebook import tqdm
device = 'cuda' if cuda.is_available() else 'cpu'

data_loc = 'Data/aclImdb'
def load_data(data_loc, split):
    X, y = [], []
    for label_name, label_id in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(data_loc, split, label_name)
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                X.append(open(os.path.join(folder, fname), encoding="utf-8").read().strip())
                y.append(label_id)
    return X, y


train_X, train_y = load_data(data_loc, "train")
test_X,  test_y  = load_data(data_loc, "test")

class MultiLabelDataset(torch.utils.data.Dataset):

    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.targets = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            ##pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

class BERTClass(torch.nn.Module):
    def __init__(self, NUM_OUT):
        super(BERTClass, self).__init__()
                   
        self.l1 = BertModel.from_pretrained("bert-base-uncased")
#       self.pre_classifier = torch.nn.Linear(768, 256)
        self.classifier = torch.nn.Linear(768, NUM_OUT)
        self.dropout = torch.nn.Dropout(0.3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
#         pooler = self.pre_classifier(pooler)
#         pooler = torch.nn.Tanh()(pooler)
#         pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def train(model, training_loader, optimizer):
    model.train()
    losses = 0;
    for data in tqdm(training_loader):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses/len(training_loader)
    
def validation(model, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for data in tqdm(testing_loader):
            targets = data['targets']
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            outputs = torch.softmax(outputs, dim=1).cpu().detach()
            fin_outputs.extend(outputs)
            fin_targets.extend(targets)
    return torch.stack(fin_outputs), torch.stack(fin_targets)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(train_X[5])

results = tokenizer(
            train_X[5],
            None,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
print(results)

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
NUM_OUT = 2
LEARNING_RATE = 2e-05

training_data = MultiLabelDataset(train_X, train_y, tokenizer, MAX_LEN)
test_data = MultiLabelDataset(test_X, test_y, tokenizer, MAX_LEN)

train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }    

training_loader = torch.utils.data.DataLoader(training_data, **train_params)
testing_loader = torch.utils.data.DataLoader(test_data, **test_params)

model = BERTClass(NUM_OUT)
model.to(device)    

optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    loss = train(model, training_loader, optimizer)
    print(f'Epoch: {epoch}, Avg Loss:  {loss}')  
    guess, targs = validation(model, testing_loader)
    _, guesses = torch.max(guess, dim=1)
    ##targets = torch.max(targs, dim=1)
    print('accuracy on test set {}'.format(accuracy_score(targs.cpu().numpy(), guesses.cpu().numpy())))
        
