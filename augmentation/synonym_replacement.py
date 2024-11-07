import argparse
import os
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

from utils.util import set_seed


def main(arg):
    # paramterts
    SEED = arg.seed
    MODEL_ID = arg.model_id
    NUM_LABELS = arg.n
    TOP_K = arg.k

    # random seeding
    set_seed(SEED)

    # data loading
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")

    data = pd.read_csv(os.path.join(DATA_DIR, "droped_train_BT.csv"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, num_labels=NUM_LABELS)

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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-s",
        "--seed",
        default=456,
        type=int,
        help="setting random seed (default: 456)",
    )
    args.add_argument(
        "-m",
        "--model_id",
        default="FacebookAI/xlm-roberta-large",
        type=str,
        help="hugging face model id (default: FacebookAI/xlm-roberta-large)",
    )
    args.add_argument(
        "-n",
        "--num_labels",
        default=7,
        type=int,
        help="the number of labels to predict (default: 7)",
    )
    args.add_argument(
        "-k",
        "--top_k",
        default=3,
        type=int,
        help="the number of candidates for synonym replacement (default: 3)",
    )

    arg = args.parse_args()
    main(arg)
