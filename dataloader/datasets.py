import torch
from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=True,
        )
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        inputs["token_type_ids"] = [label] * len(inputs["token_type_ids"])
        return inputs
