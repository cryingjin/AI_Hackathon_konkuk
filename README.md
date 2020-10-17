# Konkuk University AI Hackathon #1

건국대학교 2020-2학기 인공지능 수업 1차 해커톤 "네이버 영화평 감정분석" 을 수행한 repository 입니다.

## Usage

Clone

```
$ git clone https://github.com/HyunLee103/AI_Hackathon_konkuk/
```

데이터와 미리 학습된 모델 파일들을 다음과 같은 디렉토리 구조에 맞게 넣어야 합니다:

```
.
├── README.md
├── data
│   ├── test_label.pickle
│   ├── test_tokens.pickle
│   ├── train_label.pickle
│   └── train_tokens.pickle
├── data_utils.py
├── embedding_utils.py
├── features.py
├── main.py
└── model
    ├── doc2vec.model
    └── word2vec.model
```

Classifier 학습을 위해서는 다음 스크립트를 실행 해줍니다:

```
$ python main.py
```

모든 분석기에 대하여 결과를 출력해 줍니다.

### Submit

최종 평가 데이터에 대한 제출 결과 파일 생성을 위해서는 다음을 실행해줍니다:

```
$ python submit.py
```
