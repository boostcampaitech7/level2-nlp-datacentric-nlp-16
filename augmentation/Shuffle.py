import pandas as pd
import re
import random
from tqdm import tqdm
import argparse


def shuffle_words_excluding_particles(text):
    """
    조사를 제외하고 단어 순서를 무작위로 변경하는 함수

    Args:
        text (str): 입력 텍스트

    Returns:
        str: 조사가 제외된 단어 순서가 무작위로 섞인 텍스트
    """
    # 텍스트를 단어 단위로 분할
    words = text.split()
    
    # 조사와 비조사 단어 분리
    particles = []  # 조사 단어를 저장할 리스트
    non_particles = []  # 조사가 아닌 단어를 저장할 리스트

    for word in words:
        if re.search(r'(은|는|이|가|을|를|에|와|과|도|에서|으로|로|에게|한테|께)$', word):
            particles.append(word)
        else:
            non_particles.append(word)

    # 조사가 아닌 단어들의 순서를 무작위로 섞기
    random.shuffle(non_particles)
    
    # 섞은 비조사 단어와 조사 단어를 다시 원래 순서에 맞게 조합
    result = []
    non_particle_idx = 0
    
    for word in words:
        if word in particles:
            result.append(word)
        else:
            result.append(non_particles[non_particle_idx])
            non_particle_idx += 1
    
    return ' '.join(result)


def main(input_file, output_file):
    # 데이터 로드
    data = pd.read_csv(input_file)

    tqdm.pandas()

    data['shuffle_text'] = data['cleaned_text'].progress_apply(shuffle_words_excluding_particles)

    data.to_csv(output_file, index=False)
    print(f"Shuffled dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle words excluding particles.")
    parser.add_argument(
        "-i", "--input_file",
        default='bestdata_BT.csv',
        type=str,
        help="Input CSV file containing the text data (default: bestdata_BT.csv)"
    )
    parser.add_argument(
        "-o", "--output_file",
        default='bestdata_BT_shuffled.csv',
        type=str,
        help="Output CSV file for the shuffled text data (default: bestdata_BT_shuffled.csv)"
    )
    
    args = parser.parse_args()
    main(args.input_file, args.output_file)
