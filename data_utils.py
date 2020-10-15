import os
import joblib
from konlpy.tag import Okt


def load_data(base_dir):
    """
    파일을 읽어서 빈 데이터 제거 후 sentences 와 labels 로 분리하여 리턴

    Args:
        base_dir(str): 파일 폴더 경로

    Returns:
        tuple(list(list(str)), list(int), list(list(str)), list(int)): 
        (train_tokens, train_labels, test_tokens, test_labels) 형태의 데이터
    """
    train_tokens = joblib.load(os.path.join(base_dir, 'train_tokens.pickle'))
    train_labels = joblib.load(os.path.join(base_dir, 'train_label.pickle'))

    test_tokens = joblib.load(os.path.join(base_dir, 'test_tokens.pickle'))
    test_labels = joblib.load(os.path.join(base_dir, 'test_label.pickle'))

    return train_tokens, list(train_labels), test_tokens, list(test_labels)


def tokenize(sentence):
    """
    형태소를 분석하여 토큰화 후 [명사, 형용사, 부사] 만 리스트로 리턴

    Args:
        sentence (str): 문장

    Returns:
        list(str): 형태소로 쪼개진 리스트
    """
    stopwords = [
        '의', '가', '이', '은', '들', '을', '는', '좀', '잘', '걍',
        '과', '도', '를', '으로', '자', '에', '와', '한', '하다', '에서',
        '되', '그', '수', '나', '것', '하', '있', '보', '주', '아니', '등',
        '같', '때', '년', '가', '한', '지', '오', '말', '일', '다', '이다'
    ]

    tagger = Okt()
    morphs = tagger.pos(sentence, norm=True, stem=True)

    using_pos = ['Noun', 'Adjective', 'Adverb']

    return [
        morph[0] for morph in morphs
        if morph[1] in using_pos and morph[0] not in stopwords
    ]


def positive_negative_split(tokens_list, labels):
    """
    전체 데이터를 긍정/부정 레이블에 따라 분리

    Args:
        tokens (list(list(str))): 토큰화된 문장 리스트들
        labels (list(int)): 레이블 리스트

    Returns:
        tuple(list(str), list(int)): 긍/부정 문장 및 레이블
    """
    pos_tokens_list = []
    pos_labels = []
    neg_tokens_list = []
    neg_label = []

    for tokens, label in zip(tokens_list, labels):
        if label == 0:
            neg_tokens_list.append(tokens)
            neg_label.append(label)
        else:
            pos_tokens_list.append(tokens)
            pos_labels.append(label)

    pos_doc = [' '.join(tokens) for tokens in pos_tokens_list]
    neg_doc = [' '.join(tokens) for tokens in neg_tokens_list]

    return pos_doc, pos_labels, neg_doc, neg_label
