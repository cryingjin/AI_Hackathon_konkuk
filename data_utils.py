import os
import joblib
import re
from konlpy.tag import Okt


def load_pickle_data(base_dir):
    """
    토큰화하여 저장된 데이터 파일을 불러와서 리턴

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


def load_text_data(base_dir):
    """
    파일을 읽어서 빈 데이터 제거 후 sentences 와 labels 로 분리하여 리턴

    Args:
        base_dir(str): 파일 폴더 경로

    Returns:
        tuple(list(str), list(int)): (tokens, labels) 형태의 데이터
    """
    indices = []
    sentences = []

    with open(base_dir, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            index, sentence = line.strip().split('\t')
            cleaned_sentence = re.sub('[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+', ' ', sentence).strip()

            indices.append(index)
            sentences.append(cleaned_sentence)

        return indices, sentences


def tokenize(sentence):
    """
    형태소를 분석하여 토큰화 후 [명사, 형용사, 부사] 만 리스트로 리턴

    Args:
        sentence (str): 문장

    Returns:
        list(str): 형태소로 쪼개진 리스트
    """

    tagger = Okt()
    morphs = tagger.pos(sentence)
    non_using_pos = ['Josa', 'Punctuation', 'Number', 'Modifer', 'Eomi']

    return [
        morph[0] for morph in morphs
        if morph[1] not in non_using_pos
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
