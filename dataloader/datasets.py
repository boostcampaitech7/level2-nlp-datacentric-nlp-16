import torch
from torch.utils.data import Dataset

class HeadlineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        inputs['labels'] = label
        return inputs