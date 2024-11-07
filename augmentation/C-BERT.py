import os
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

import wandb
from dataloader.datasets import MaskedDataset
from utils.util import set_seed


def main(arg):
    ## paramterts
    SEED = 456
    MODEL_ID = arg.model_id
    NUM_LABELS = arg.n
    TOP_K = arg.k
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.00001

    ## random seeding
    set_seed(SEED)

    ## data loading
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    data = pd.read_csv(os.path.join(DATA_DIR, "train_cleaned.csv"))

    ## model/tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, num_labels=NUM_LABELS)

    ## label embedding setting
    model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(NUM_LABELS, model.config.hidden_size)

    ## parameter freezing
    for name, param in model.named_parameters():
        if "token_type_embeddings" in name:
            param.requires_grad = True  ## For training label embedding
            continue
        param.requires_grad = False

    wandb.init(
        project="Level2-datacentric",
        name="c-bert",
    )

    # dataset call
    train_dataset = MaskedDataset(data["text"].tolist(), data["target"].tolist(), tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # trainging arguments setting
    training_args = TrainingArguments(
        logging_dir="./logs",
        output_dir="./checkpoints",
        report_to="wandb",
        logging_steps=100,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # trainer setting
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # training
    trainer.train()
    wandb.finish()

    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        top_k=TOP_K,
    )
    outputs = []
    for row in tqdm(data.itertuples()):
        text_split = row.text.split()

        ## the number of tokens to be masked varies depending on the split length
        if len(text_split) > 5:
            indices = random.sample(range(len(text_split)), k=3)
        elif len(text_split) > 2:
            indices = random.sample(range(len(text_split)), k=1)
        else:  ## if length is less than or equal to 2, nothing will be changed
            indices = []

        for idx in indices:
            text_split[idx] = tokenizer.mask_token
            input = " ".join(text_split)
            result = fill_mask(input).pop()  ## select lowest priority for getting diversity of expression
            result = result["token_str"]
            text_split[idx] = result

        output = " ".join(text_split)
        output.replace("</s>", "")
        output.replace("<s>", "")
        outputs.append(output)
