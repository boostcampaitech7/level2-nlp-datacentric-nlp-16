import argparse
import os

import pandas as pd
import transformers

from utils.clean_text import calculate_ratio, denoise_text
from utils.util import set_seed

transformers.logging.set_verbosity_error()


def main(arg):
    """
    Clean dataset using Korean ratio and LM

    Firstly, detect noisy texts by Korean character ratio
    Nextly, denoise the text using LM with prompt engineering

    Args:
        SEED (int): random seed number
        MODEL_ID (str): huggingface model id
        KR_UB (float): Korean ratio upper bound for cleanable noisy texts group
        KR_LB (float): Korean ratio lower bound for cleanable noisy texts group
    """
    ## parameters
    SEED = arg.seed
    MODEL_ID = arg.model_id
    KR_UB = arg.kr_ub
    KR_LB = arg.kr_lb

    ## random seeding
    set_seed(SEED)

    ## data loading
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    ## text ratio
    res = data["text"].apply(lambda x: calculate_ratio(x))
    korean_ratio = res[1]

    idx_korean = (korean_ratio >= KR_LB) & (korean_ratio < KR_UB)
    idx = idx_korean
    cleanable_data = data[idx]

    ## denoise
    with open(os.path.join(BASE_DIR, "prompt.txt"), "r") as f:
        template = f.read()
    output_txts = denoise_text(
        texts=cleanable_data["text"].tolist(),
        model_id=MODEL_ID,
        template=template,
    )

    data.loc[idx, "text"] = output_txts

    ## remove not-cleanable text
    idx_korean = korean_ratio < KR_LB
    idx = idx_korean
    data = data[~idx]

    data.to_csv(os.path.join(DATA_DIR, "train_cleaned.csv"), index=False)


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
        default=None,
        type=str,
        help="hugging face model id (default: None)",
    )
    args.add_argument(
        "-ku",
        "--kr_ub",
        default=0.75,
        type=float,
        help="upper bound for korean ratio in text (default: 0.75)",
    )
    args.add_argument(
        "-kl",
        "--kr_lb",
        default=0.5,
        type=float,
        help="lower bound for korean ratio in text (default: 0.5)",
    )

    arg = args.parse_args()
    main(arg)
