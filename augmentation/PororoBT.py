import argparse
import pandas as pd
from tqdm import tqdm
from pororo import Pororo
import torch


def augment_text_back_translation(text, mt):
    """
    역번역을 통해 텍스트를 증강하는 함수

    Args:
        text (str): 입력 텍스트
        mt (Pororo): Pororo 번역 모델 인스턴스

    Returns:
        str: 증강된 텍스트
    """
    # 영어로 번역
    translated_en = mt(text, src="ko", tgt="en")
    # 다시 한국어로 번역
    back_translated_ko = mt(translated_en, src="en", tgt="ko")
    return back_translated_ko


def main(input_file, output_file):
    # 데이터셋 로드
    df = pd.read_csv(input_file)

    # GPU가 사용 가능한지 확인하고, 있으면 GPU에 올리기
    if torch.cuda.is_available():
        print("Using GPU (CUDA) for processing")
    else:
        print("Using CPU for processing")

    # Pororo translation 모델 로드
    mt = Pororo(task="translation", lang="multi")

    tqdm.pandas(desc="Augmenting Data")
    df['BT_text'] = df['text'].progress_apply(lambda x: augment_text_back_translation(x, mt))

    # 증강된 데이터셋 저장
    df.to_csv(output_file, index=False)
    print(f"Augmented dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text augmentation using back translation.")
    parser.add_argument(
        "-i", "--input_file",
        default='restored_train_v2.csv',
        type=str,
        help="Input CSV file containing the text data (default: restored_train_v2.csv)"
    )
    parser.add_argument(
        "-o", "--output_file",
        default='restored_train_v2_BT.csv',
        type=str,
        help="Output CSV file for the augmented text data (default: restored_train_v2_BT.csv)"
    )
    
    args = parser.parse_args()
    main(args.input_file, args.output_file)
