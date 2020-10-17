import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


def mk_stats(pos, neg, token):
    tfidfv = TfidfVectorizer(max_features=3000).fit(pos)
    score = {
        key: value for key, value in zip(
            sorted(tfidfv.vocabulary_, key=lambda x: tfidfv.vocabulary_[x]),
            tfidfv.transform(pos).toarray().sum(axis=0)
        )
    }
    top_pos = sorted(score, key=lambda x: score[x], reverse=True)[:500]

    tfidfv = TfidfVectorizer(max_features=3000).fit(neg)
    score = {
        key: value for key, value in zip(
            sorted(tfidfv.vocabulary_, key=lambda x: tfidfv.vocabulary_[x]),
            tfidfv.transform(neg).toarray().sum(axis=0)
        )
    }
    top_neg = sorted(score, key=lambda x: score[x], reverse=True)[:500]

    pos_only = set(top_pos) - (set(top_pos) & set(top_neg))
    neg_only = set(top_neg) - (set(top_pos) & set(top_neg))

    pos_cnt = [len(set(i) & pos_only) for i in token]
    neg_cnt = [len(set(i) & neg_only) for i in token]

    ratio = (np.array(pos_cnt)+1)/(np.array(neg_cnt)+1)
    return np.concatenate((np.array(pos_cnt).reshape(-1, 1), np.array(neg_cnt).reshape(-1, 1), ratio.reshape(-1, 1)), axis=1)


"""
input_data : raw_data에 결측값, duplicate 전처리만 한 거
pos : 긍정문서 집합
neg : 부정문서 집합
token : token
"""


def mk_feature(input_data, token):
    pos = joblib.load('data/neg.pickle')
    neg = joblib.load('data/neg.pickle')

    # 1. 순서대로 !!! 이랑 '흠' 있으면 1 없으면 0
    has_admiration = np.array([
        1 if '!!!' in data else 0
        for data in input_data
    ])
    has_hmm = np.array([1 if '흠' in data else 0 for data in input_data])

    posneg_stat = mk_stats(pos, neg, token)

    tem = np.concatenate(
        (
            posneg_stat,
            has_admiration.reshape(-1, 1),
            has_hmm.reshape(-1, 1)
        ),
        axis=1
    )

    scale = MinMaxScaler()
    return scale.fit_transform(tem)
