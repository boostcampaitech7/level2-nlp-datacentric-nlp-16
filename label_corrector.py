import argparse
import os

import pandas as pd
import torch
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from utils.util import set_seed


def main(arg):
    """
    Clean label nosie using cleanlab

    Args:
        SEED (int): random seed number (default: 456)
        MODEL_ID (str): huggingface model id (default: klue/roberta-base)
        MAX_ITER (int): max iteration for logistic regression (default: 400)
        K (int): number of folds for cross validation (default: 5)
    """
    SEED = arg.seed
    MODEL_ID = arg.model_id
    MAX_ITER = arg.max_iter
    K = arg.k

    ## random seeding
    set_seed(SEED)

    ## data loading
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    ## model loading
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## embedding
    tokenized = tokenizer(
        data["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    inputs = {key: val.to(device) for key, val in tokenized.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    text_emb = outputs.last_hidden_state.mean(dim=1)
    text_emb = text_emb.cpu().numpy()

    ## clean label
    model = LogisticRegression(max_iter=MAX_ITER)
    cl = CleanLearning(model, cv_n_folds=K)

    label_issues = cl.find_label_issues(X=text_emb, labels=data["target"].values)
    idx = label_issues["is_label_issue"] == True
    data.loc[idx, "target"] = label_issues.loc[idx, "predicted_label"]

    data.to_csv(os.path.join(DATA_DIR, "train_corrected.csv"), index=False)


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
        default="klue/roberta-base",
        type=str,
        help="hugging face model id (default: klue/roberta-base)",
    )
    args.add_argument(
        "-mi",
        "--max_iter",
        default=400,
        type=int,
        help="max iteration for logistic regression (default: 400)",
    )
    args.add_argument(
        "-k",
        "--k",
        default=5,
        type=int,
        help="number of folds for cross validation (default: 5)",
    )

    arg = args.parse_args()
    main(arg)
