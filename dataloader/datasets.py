import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    """
    Dataset class for C-BERT
    Save tokenized input text and Update token type index to label index for label embedding

    Args:
        texts (List[str]): input texts
        labels (List[int]): corresponding target labels for input texts
        tokenizer: tokenizer for input texts
        max_length: max token length for text (default: 32)
    """

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



class MixupBERTDataset(Dataset):
    """
    Dataset class for applying Mixup to BERT tokenized data.
    
    This class creates a dataset for BERT models that includes Mixup augmentation within
    groups of data points with the same target labels. Mixup is a data augmentation 
    technique that interpolates between two training examples to create a new example.

    Args:
        data (DataFrame): A DataFrame containing the input data. It must have 'text' and 'target' columns.
        tokenizer: The tokenizer used to tokenize the input text.
        alpha (float, optional): The parameter for the Beta distribution used to sample the mixup ratio λ (default: 0.5).

    Attributes:
        grouped_data (dict): A dictionary where each key is a target label, and the value is a dictionary 
                             containing 'inputs' (tokenized text) and 'labels' for that label group.
        alpha (float): The mixup parameter controlling the strength of interpolation.

    Methods:
        __getitem__(self, idx):
            Retrieves a mixed example from the dataset. Selects two examples from the same label group 
            and applies Mixup augmentation to create a new example.

        __len__(self):
            Returns the total number of examples in the dataset (sum of all examples in all label groups).
    """

    def __init__(self, data, tokenizer, alpha=0.5):
        self.tokenizer = tokenizer
        self.alpha = alpha  # mixup parameter
        
        # Grouping data by target labels
        self.grouped_data = {}
        for i in range(len(data)):
            label = data['target'].iloc[i]
            if label not in self.grouped_data:
                self.grouped_data[label] = {'inputs': [], 'labels': []}
            tokenized_input = tokenizer(data['text'].iloc[i], padding='max_length', truncation=True, return_tensors='pt')
            self.grouped_data[label]['inputs'].append(tokenized_input)
            self.grouped_data[label]['labels'].append(torch.tensor(label))

    def __getitem__(self, idx):
        """
        Retrieves a mixed example from the dataset using Mixup.

        Args:
            idx (int): The index of the item to be retrieved.

        Returns:
            dict: A dictionary with keys 'input_ids', 'attention_mask', and 'labels', 
                representing the mixed input IDs, attention masks, and labels, respectively.
        """
        # Randomly select a target group
        target_group = random.choice(list(self.grouped_data.keys()))
        group_data = self.grouped_data[target_group]
        
        # Select two different examples without replacement from the group
        idx1, idx2 = random.sample(range(len(group_data['inputs'])), 2)

        input1 = group_data['inputs'][idx1]
        label1 = group_data['labels'][idx1]
        input2 = group_data['inputs'][idx2]
        label2 = group_data['labels'][idx2]

        # Generate Mixup lambda value
        lam = np.random.beta(self.alpha, self.alpha)

        # Apply Mixup to input_ids and attention_mask
        mixed_input_ids = lam * input1['input_ids'] + (1 - lam) * input2['input_ids']
        mixed_attention_mask = lam * input1['attention_mask'] + (1 - lam) * input2['attention_mask']

        # Apply Mixup to labels
        mixed_labels = lam * label1 + (1 - lam) * label2

        return {
            'input_ids': mixed_input_ids.squeeze(0).long(),
            'attention_mask': mixed_attention_mask.squeeze(0).long(),
            'labels': mixed_labels.long()  # Labels are typically integers, so conversion is applied
        }

    def __len__(self):
        """
        Returns:
            int: The total number of examples in the dataset.
        """
        return sum(len(group['inputs']) for group in self.grouped_data.values())