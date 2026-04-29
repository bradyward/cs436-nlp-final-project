import os
import numpy as np
from sklearn.metrics import accuracy_score

from transformers import BertTokenizer, BertModel

import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_LOC = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'aclImdb')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'bert_model')

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
NUM_OUT = 2
LEARNING_RATE = 2e-05


def load_data(data_loc, split):
    X, y = [], []
    for label_name, label_id in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(data_loc, split, label_name)
        for fname in os.listdir(folder):
            if fname.endswith(".txt"):
                X.append(open(os.path.join(folder, fname), encoding="utf-8").read().strip())
                y.append(label_id)
    return X, y


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
            truncation=True,
            return_token_type_ids=True
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }


class BERTClass(torch.nn.Module):
    def __init__(self, num_out):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_out)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler = output_1[0][:, 0]
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return self.softmax(output)


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)


def train(model, training_loader, optimizer):
    model.train()
    losses = 0
    for data in tqdm(training_loader, desc="Training"):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(training_loader)


def validation(model, testing_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for data in tqdm(testing_loader, desc="Validating"):
            targets = data['targets']
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            fin_outputs.extend(torch.softmax(outputs, dim=1).cpu().detach())
            fin_targets.extend(targets)
    return torch.stack(fin_outputs), torch.stack(fin_targets)


def score_text(text: str, model: 'BERTClass', tokenizer: 'BertTokenizer') -> float:
    """Return positive sentiment probability (0.0-1.0) for a single text."""
    model.eval()
    inputs = tokenizer(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=True
    )
    ids = torch.tensor([inputs['input_ids']], dtype=torch.long).to(device)
    mask = torch.tensor([inputs['attention_mask']], dtype=torch.long).to(device)
    token_type_ids = torch.tensor([inputs['token_type_ids']], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(ids, mask, token_type_ids)
    return float(output[0][1].cpu())


def load_model(model_path: str) -> tuple:
    """Load saved model and tokenizer. Returns (model, tokenizer)."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClass(NUM_OUT)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    train_X, train_y = load_data(DATA_LOC, "train")
    test_X, test_y = load_data(DATA_LOC, "test")
    print(f"  Train: {len(train_X)} | Test: {len(test_X)}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    training_data = MultiLabelDataset(train_X, train_y, tokenizer, MAX_LEN)
    test_data = MultiLabelDataset(test_X, test_y, tokenizer, MAX_LEN)

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testing_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BERTClass(NUM_OUT)
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss = train(model, training_loader, optimizer)
        print(f'Epoch {epoch + 1}/{EPOCHS} | Avg Loss: {loss:.4f}')

        guess, targs = validation(model, testing_loader)
        _, guesses = torch.max(guess, dim=1)
        acc = accuracy_score(targs.cpu().numpy(), guesses.cpu().numpy())
        print(f'  Accuracy: {acc:.4f}')

        epoch_path = os.path.join(MODEL_DIR, f'bert_epoch_{epoch + 1}.pt')
        torch.save(model.state_dict(), epoch_path)
        print(f'  Saved checkpoint: {epoch_path}')

    final_path = os.path.join(MODEL_DIR, 'bert_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f'Training complete. Final model saved: {final_path}')
