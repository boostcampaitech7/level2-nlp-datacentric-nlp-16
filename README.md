# :rocket: Data-Centric Topic Classification

## :closed_book: 프로젝트 개요
본 프로젝트는 **Topic Classification** 문제를 **Data-Centric한 접근**을 통해 해결하는 것을 주제로 하고 있습니다.

현업에서 데이터의 중요도는 매우 높습니다. 하지만, ML/DL 모델에 비해 **데이터에 관한 연구는 활발히 이루어지지 않고 있습니다**. 본 프로젝트는 이러한 흐름에서 벗어나 **모델에 대한 수정 없이** Data-Centric한 접근 만으로 모델의 성능을 극대화시키고, 해당 과정에서 **데이터의 품질을 개선할 수 있는 다양한 방안에 대해 탐구하는 것**을 목표로 하고 있습니다.

이러한 접근 방식은 현업에서의 복잡하고 잘 정제되지 않은 데이터들을 적절히 처리하는 일에 솔루션을 제공할 수 있습니다.

## :closed_book: 프로젝트 요약
- **Text denoise** 단계 별 세세한 instruction을 **prompt**로 입력하여 **LM 기반의 denoising** 수행
- **Labeling error를 교정**하기 위해, **text embedding**과 함께 각종 **ML model** 활용
- Back translation, Mix-up, C-BERT, Word Random Shuffle 등 **부족한 train data size를 보완**하기 위해 각종 **augmentation method** 시도

## :family_man_man_boy_boy: 멤버 소개
|강경준|김재겸|원호영|유선우|
|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/da281cf3-b2cc-4022-ae9e-68ed6c174cd7" alt="KKJ" width="1000" height="250"> | <img src="https://github.com/user-attachments/assets/5ecd2475-eb8c-4662-bed0-59ae27cc2e0c" alt="KJK" width="1000" height="250"> | <img src="https://github.com/user-attachments/assets/e02832ae-f1fa-4b4f-a7e9-fcc511c727e7" alt="WHY" width="1000" height="250"> | <img src="https://github.com/user-attachments/assets/d58b7d68-ef8e-4a12-ae6e-0d50458e0c5b" alt="YSW" width="1000" height="250"> |

## :balance_scale: 역할 분담
|팀원| 역할 |
|:---:| --- |
| 강경준 | noisy text detection, text denoising, c-bert, code 관리 |
| 김재겸 | text denoising, noisy text detection, data relabeling, mix-up |
| 원호영 | augmentation, text denoising, back translation, word random shuffle |
| 유선우 | EDA, analysis for noise pattern, text denoising |

## :computer: 개발/협업 환경
- **컴퓨팅 환경**
	- V100 서버 (VS code와 SSH로 연결하여 사용)
- **협업 환경**
  	- ![notion](https://img.shields.io/badge/Notion-FFFFFF?style=flat-square&logo=Notion&logoColor=black) ![github](https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white) ![WandB](https://img.shields.io/badge/WeightsandBiases-FFBE00?style=flat-square&logo=WeightsandBiases&logoColor=white)
- **의사소통**
  	- ![zoom](https://img.shields.io/badge/Zoom-0B5CFF?style=flat-square&logo=Zoom&logoColor=white)

## :bookmark_tabs: 데이터 설명
- **데이터 설명**
  - Train / Test (**2,800** / **30,000**)
  - 구성
	  - ID : 각 데이터 샘플 ID
    - **text** : 기사 제목
    - **target** : 기사 분류 / 정수형 인코딩
      - 생활문화, 스포츠, 세계, 정치, 경제, IT과학, 사회의 7가지 주제 중 하나
  - 특징
    - text, target에 **noise 포함**
    - text 중 일부를 **다른 ascii 코드**로 변경
    - target 중 일부 **임의로 변경**

## :card_index_dividers: 프로젝트 구조
```
. level2-datacentric-nlp-16
├─ .github
├─ data
│  ├─ train.csv
│  └─ test.csv
├─ dataloader
│  └─ datasets.py
├─ augmentation
│  ├─ C-BERT.py
│  ├─ PororoBT.py
│  ├─ shuffle.py
│  └─ synonym_replacement.py
├─ prompts
│  ├─ prompt_gemma.py
│  └─ prompt_llama.py
├─ utils
│  ├─ clean_text.py
│  └─ util.py
├─ .flake8
├─ .gitignore
├─ .gitmessage.txt
├─ .pre-commit-config.yaml
├─ README.md
├─ requirements.txt
├─ baseline_code.ipynb
├─ clean.py
└─ label_corrector.py
```

## :movie_camera: 실행 코드

### clean.py

```bash
# text denoising
## -s : seed number setting
## -m : huggingface model id
## -ku : korean ratio upper bound
## -kl : korean ratio lower bound
python3 clean.py -s 456 -m aifeifei798/Meta-Llama-3.1-8B-Instruct -ku 0.75 -kl 0.5
```

### label_corrector.py

```bash
# label denoising
## -s : seed number setting
## -m : huggingface model id
## -mi : max iteration for logistic regression
## -k : number of folds for cross validataion
python3 label_corrector.py -s 456 -m klue/roberta-base -mi 400 -k 5
```

### C-BERT.py

```bash
# C-BERT augmentation
## -s : random seed number
## -m : huggingface model id
## -n : the number of labels to predict
## -k : the number of candidates for synonym replacements
## -e : epoch size
## -b : batch size
## -lr : learning rate for training C-BERT
## -w : weight decay for learning rate scheduler
python3 C-BERT.py -s 456 -m FacebookAI/xlm-roberta-large -n 7 -k 3 -e 10 -b 16 -lr 0.001 -w 0.0001
```

### synonym_replacement.py

```bash
# synonym replacement augmentation
## -s : random seed number
## -m : huggingface model id
## -n : the number of labels to predict
## -k : the number of candidates for synonym replacements
python3 synonym_replacement.py -s 456 -m FacebookAI/xlm-roberta-large -n 7 -k 3
```

### PororoBT.py

```bash
# back translation augmentation
## -i : "input.csv" name 
## -o : "output.csv" name
python3 PororoBT.py -i train.csv -o ouput.csv
```

### Shuffle.py

```bash
# Random word shuffle augmentation
## -i : "input.csv" name 
## -o : "output.csv" name
python3 Shuffle.py -i train.csv -o ouput.csv
```

## :book: 프로젝트 수행 결과
- **Text Noise Detection**
	- **목적**
		- data 상에는 수정하지 않아도 될 만큼 깔끔한 text와 어느 정도 수정이 가능한 text, 수정할 수 없는 정도로 손상된 text 등 여러 유형의 text가 존재
		- text에 대한 **손상 정도를 정의**하여, **수정할 text를 선별**하는 과정 필요
	- **한국어 비율**
	    - 한글 문자가  영어, 특수 문자, 공백 및 숫자 등으로 대체되는 방식의 noise
	    - 손상이 큰 text는 **text 내 한글 비율이 낮을 것**으로 예상됨
	    - 따라서, 한글 비율을 기준으로  구간을 나누어 **`normal data`, `cleanable data`, `not-cleanable data`** 분류
	    - ***normal data***
	      - 전혀 손상되지 않았거나, 수정하지 않아도 괜찮은 수준
	      - 예시 (한글 및 공백 비율 **0.8 이상** 기준)
		- 페이스북 인터넷 드론 아퀼라 실물 첫 시험비행 성공
		- 해외로밍 m금폭탄 n동차단 더 빨$진다
		- 땅 파= 코l나 격리시설 탈출한 외국인 청_서 VS
	    - ***cleanable data***
	      - 약간 손상됐지만, 원형을 어느정도 추정 가능한 수준
	      - 예시 (한글 및 공백 비율 **0.8 미만 0.6 이상** 기준)
		- m 김정) 자주통일 새,?r열1나가야1보
		- 코로나 r대^등교)모습
		- 문대통령 김정*m트/프7 YTD 조속히H끝내고 A다고!,p2합
	    - ***not-cleanable data***
	      - 손상이 너무 커, 원형을 추정할 수 없는 수준
	      - 예시 (한글 및 공백 비율 **0.6 미만** 기준)
		- E달A](j상ZwQ선 일*77아-는데… nfD편
		- .달 CES %굴#N바@은^새a|더o폰I중저o폰O rb
		- 여행^식e한$8수&mT30,_Y기! 사진# 이마진 프<스

- **Text Denoising**
  - **목적**
    - text에 적용된 **noise 규칙을 분석**하고, 이를 기반으로 **원형을 복구하는 것을 목적**으로 함
    - text의 원형에 대한 데이터를 활용한 **학습을 진행할 수 없으**므로, **LLM에 prompt engineering을 적용**하는 방안 활용
  - **모델 선정**
    - 선정 기준
      - noise가 섞인 문장을 복원하는 작업에는 **높은 수준의 추론 능력 요구**됨
      - 하지만, **제한된 컴퓨팅 리소스** 내에서 기대할 수 있는 모델의 추론 능력에는 한계가 있음
      - 따라서, **작은 크기로 최대한의 효용**을 낼 수 있는 모델을 기준으로 선정
    - ***Llama***
      - **scaling law 기반**으로 model size에 적합한 dataset size를 통해, **제한된 모델 크기 내에서 최고의 효율성** 달성
      - **`Llama - 8B`** 모델 활용
    - ***Gemma***
      - **scaling law 기반**의 적절한 dataset size에 더불어, 작은 모델에 큰 모델의 지식을 전달하는 **지식 증류 학습 기술**을 통해 작은 크기의 모델에서도 좋은 성능을 달성
      - **`Gemma - 9B`** 모델 활용
  - **Prompt Engineering**
    - ***Few shot***
      - **`Llama - 8B`** 모델 활용
      - 문장 복원에 대한 **예시 기반**의 **질문과 답변 쌍**으로만 prompt를 구성
      - 결과
        - **Llama model** 활용
        - **Accuracy** : 0.7205 / **F1-score** : 0.7041
        - CoT 기반의 prompt에 비해 **부족한 성능**
        - **자의적으로 수정**하는 경우 대부분
        - **손상이 덜한 텍스트**에 오히려 **손상을 주는 경우**도 발생
    - ***Chain of Thoughts***
      - 문장의 복원 단계 및 규칙을 세부적으로 나누어 지시
      - ***Llama model base***
        - **`Llama - 8B`** 모델 활용
        - 노이즈 패턴 설명⋅복원 규칙 제시⋅예시⋅복원 요청문의 **4단계로 구성**
        - 결과
          - 문장의 **완결성 높음**
          - 다소 **자의적인 수정**을 하는 경우 여전히 존재
          - **Accuracy** : 0.8277 / **F1-score** : 0.8247
      - ***Gemma model base***
        - **`Gemma - 9B`** 모델 활용
        - 복원 조건⋅복원 단계⋅복원 예시⋅복원 요청문의 **4단계로 구성**
        - 결과
          - 조건, 예시, 요청의 **3단계 구성** 보다, **복원 단계를 추가**을 때 **복원 조건 반영 능력 향상**
          - 하나의 예시만 주었을 때 보다 **여러 개의 예시**를 주었을 때 다양한 케이스에 대한 **일관된 복원 능력** 향상에 도움 됨
          - **Accuracy** : 0.8293 / **F1-score** : 0.8257
    - 결론
      - 단순히 **예시를 연속적으로 제시**하는 것 보다, **세부적으로 과정을 설명하는 것**이 모델에게도 도움이 됨
      - 다만, 규칙을 설정해도 **규칙이 완벽하게 지켜지지는 않으므로**, 규칙에 대한 **적절한 예시도 함께 제시**하는 게 좋음
	
- **Label Denoising**
  - **Denoising Tool**
    - ***CleanLab***
      - DL기반의 **text embedding** 모델에 **linear layer를 추가**하고 해당 layer에 대한 학습을 통해, **classification task** 수행
        - **학습 코드의 용이함**과 **CleanLab과의 호환성**을 고려하여, **logistic regression**을 적용하는 방식으로 **linear layer 대체**
        - logistic regression에서 더욱 확장하여, SVM⋅Random Forest 등 **다양한 ML 모델 적용** 및 **앙상블**
      - 각 label 별 **predicted probability**에 대해 **threshold**를 설정하고, 해당 **threshold 이하일 경우 labeling error**로 판단
        - 일반적으로 threshold는 각 label로 예측된 데이터에 대해 **predicted probability의 평균값** 활용
        - **Over-correcting을 방지**하기 위하여, **Mis-labeling data의 개수**에 맞춰 **threshold 조정** 시도
    - 결과
      - ***logistic regression***
        - **Accuracy** : 0.8237 / **F1-score** : 0.8218
      - ***ensemble***
        - **모델 구성** : logistic regression, randomforestclassifier, SVM
        - **Accuracy** : 0.8266 / **F1-score** : 0.8241
	
- **Augmentation**
  - **C-BERT**
    - BERT 기반의 **Masked LM**을 활용한 유사어 대체 증강 기법
    - BERT 모델의 input에 **label embedding**을 추가하여, **label 정보를 반영한 증강** 구현
    - 결과
      - train data 상의 noise로 인해, **label embedding에 대한 적절한 학습이 어려움**
      - <MASK> 토큰을 각종 **특수문자로 예측**하는 경우 발생
  - **유의어 대체**
    - Masked LM을 활용한 **유사어 대체 증강** 기법
    - 결과
      |  | **Accuracy** | **F1-Score** |
      |--|----------|----------|
      | **증강 전** | 0.8275 | 0.8246 |
      | **증강 후** | 0.8297 | 0.8269 |
  - **Back Translation**
    - 한국어를 영어로 번역한 뒤 다시 한국어로 번역하여, **번역 과정에서 발생하는 언어 표현의 다양성** 활용
    - 번역 모델의 활용에 있어서 **번역 자체의 품질은 매우 중요**
    - **한국어 자연어 처리에 특화**되어, 한국어 번역 관련해서도 좋은 성능을 보여주는 **Kakao brain/Pororo** 자연어 처리 프레임워크의 번역 모델 활용
    - 결과
      |  | **Accuracy** | **F1-Score** |
      |--|----------|----------|
      | **증강 전** | 0.7584 | 0.7546 |
      | **증강 후** | 0.7917 | 0.7836 |
      - 번역 과정에서, **기사 제목**에서 주로 사용하는 **문체 탈피**
  - **Random word Shuffle**
    - Text에 대해 **조사를 제외**한 **단어 순서 임의로 변경**하여 데이터 증강
    - 결과
      |  | **Accuracy** | **F1-Score** |
      |--|----------|----------|
      | **증강 전** | 0.8110 | 0.8050 |
      | **증강 후** | 0.8084 | 0.8021 |
      - 임의 변경으로 인해, 문장의 **완결성 보존에 한계**
      - 기존 표현과 비교해, **다양성 확보**에 크게 도움이 되지는 않음
  - **Mix-Up**
    - **Text embedding 기준**으로 **선형 보간** 데이터 생성을 통해 증강
    - **여러 label**에 대한 데이터를 **교차로 mix**할 경우 **label 배정에 문제**가 있으므로, **각 label 별로 적용**
    - 결과
      |  | **Accuracy** | **F1-Score** |
      |--|----------|----------|
      | **증강 전** | 0.8266 | 0.8241 |
      | **증강 후** | 0.8185 | 0.8173 |

- **최종 제출 모델**
  - Text denoising 과정에서 발생할 수 있는 **원본 훼손에 대비**하여, denoising 후 **원본 텍스트에 증강**하는 방식 활용
  - Llama, Gemma model 각각 **denoising 결과에 차이**가 있으므로, 각각의 결과를 **모두 활용**
  - 기타 성능 향상을 확인할 수 있었던 **back translation**, **유의어 대체** 활용
  - 결과
    - **Accuracy** : 0.8401 / **F1-score** : 0.8365
