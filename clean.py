import os
import argparse

import pandas as pd
import transformers

from utils.util import set_seed
from utils.clean_text import calculate_ratio, denoise_text

transformers.logging.set_verbosity_error()

def main(arg):
    ## parameters
    SEED = arg.seed
    KR_UB = arg.kr_ub
    KR_LB = arg.kr_lb
    SP_UB = arg.sp_ub
    SP_LB = arg.sp_lb

    ## random seeding
    set_seed(SEED)

    ## data loading
    data = pd.read_csv(os.path.join('./data', 'train.csv'))

    ## text ratio
    res = data['text'].apply(lambda x: pd.Series(calculate_ratio(x)))
    special_ratio = res[0]
    korean_ratio = res[1]

    idx_korean = (korean_ratio >= KR_LB) & (korean_ratio < KR_UB)
    # idx_special = (special_ratio <= SP_UB) & (special_ratio > SP_LB)
    idx = idx_korean # & idx_special
    cleanable_data = data[idx]

    ## denoise
    model_id = "aifeifei798/Meta-Llama-3.1-8B-Instruct"
    with open('./codes/prompt.txt', 'r') as f:
        template = f.read()
    output_txts = denoise_text(
        texts=cleanable_data["text"].tolist(),
        model_id=model_id,
        template=template,
    )

    data.loc[idx, "text"] = output_txts

    ## remove not-cleanable text
    idx_korean = korean_ratio < KR_LB
    # idx_special = special_ratio > SP_UB
    idx = idx_korean # & idx_special
    data = data[~idx]

    data.to_csv(os.path.join('./data', 'train_cleaned.csv'), index=False)

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
    args.add_argument(
        "-su",
        "--sp_ub",
        default=0.5,
        type=float,
        help="upper bound for special symbol ratio in text (default: 0.5)",
    )
    args.add_argument(
        "-sl",
        "--sp_lb",
        default=0.25,
        type=float,
        help="lower bound for special symbol ratio in text (default: 0.25)",
    )

    arg = args.parse_args()
    main(arg)