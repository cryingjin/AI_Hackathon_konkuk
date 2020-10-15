"""
Embedding Utils
토큰화 된 문장들을 벡터화 시켜주는 함수와 그 보조함수들로 이루어져 있습니다.
"""

import numpy as np
import multiprocessing
from collections import namedtuple
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

"""
Chi-square utils
"""

# 긍정 / 혹은 부정 문장리스트

# TODO: Add docstring


def count_word(data, word):
    cnt = 0
    for sentence in data:
        if word in sentence:
            cnt += 1
    return cnt


def get_chi_square(pos_doc_list, neg_doc_list, word):
    """word 의 chi square 값을 리턴: 높을수록 긍정단어에 가깝고, 낮을수록 부정단어에 가까움

    Args:
        pos_doc_list (list(str)): 긍정문서 리스트
        neg_doc_list (list(str)): 부정문서 리스트
        word (str): 대상 단어

    Returns:
        float: 해당 단어의 카이제곱 수치
    """
    A = count_word(pos_doc_list, word)
    B = count_word(neg_doc_list, word)
    C = len(pos_doc_list) - A
    D = len(neg_doc_list) - B

    # 분모가 0이면 0을 리턴함.
    if ((A+B)*(A+C)*(B+D)*(C+D)) == 0:
        return 0
    return ((A+B+C+D)*((A*D-B*C)*(A*D-B*C))) / ((A+B)*(A+C)*(B+D)*(C+D))


"""
word2vec utils
"""


def generate_word2vec_dict(tokens, embedding_dim=128, save_model=True):
    """
    word2vec 사전을 구성

    Args:
        tokens (list(list(str))): 형태소들로 분리된 문장들
        embedding_dim (int, optional): 임베딩 차원. Defaults to 128.
        save_model (bool, optional): 훈련된 모델을 저장할지 여부. Defaults to True.

    Returns:
        dict(str: numpy.ndarray): word2vec 매트릭스 딕셔너리
    """
    model = Word2Vec(tokens, size=embedding_dim, min_count=1, sg=1)

    if save_model:
        model.save('model/word2vec.model')

    words = model.wv.index2word
    vectors = model.wv.vectors

    return {word: vector for word, vector in zip(words, vectors)}


def vectorize_matrix_with_word2vec(tokens_list, embedding_dim=128, model=None):
    """
    word2vec 문장에 의거하여 문장을 벡터화

    Args:
        tokens_list (list(list(str))): 형태소들로 분리된 문장들
        embedding_dim (int, optional): 임베딩 차원. Defaults to 128.

    Returns:
        numpy.ndarray(tokens_list_size, embedding_dim): 벡터화된 문장 리스트
    """
    size = len(tokens_list)
    matrix = np.zeros((size, embedding_dim))
    word_table = generate_word2vec_dict(tokens_list, save_model=False)

    for i, tokens in enumerate(tokens_list):
        vector = np.array([
            word_table[token] for token in tokens
            if token in word_table
        ])

        if vector.size != 0:
            final_vector = np.mean(vector, axis=0)
            matrix[i] = final_vector

    return matrix


"""
doc2vec utils
"""


def generate_doc2vec_matrix(tokens, labels):
    encoder = LabelEncoder()
    words_tags = namedtuple('TaggedDocument', ['words', 'tags'])

    str_labels = list(map(lambda x: 'pos' if x == 1 else 'neg', labels))
    tagged_document = [
        words_tags(word, tag)
        for word, tag in zip(tokens, str_labels)
    ]

    cores = multiprocessing.cpu_count()

    doc_vectorizer = Doc2Vec(
        dm=0,             # PV-DBOW / default 1
        dbow_words=1,     # w2v simultaneous with DBOW d2v / default 0
        window=5,         # distance between the predicted word and context words
        vector_size=128,  # vector size
        alpha=0.025,      # learning-rate
        seed=1234,
        min_count=1,      # ignore with freq lower
        min_alpha=0.025,  # min learning-rate
        workers=cores,    # multi cpu
        hs=1,             # hierarchical softmax / default 0
        negative=5,       # negative sampling / default 5
        epochs=20
    )

    doc_vectorizer.build_vocab(tagged_document)

    for epoch in range(10):
        doc_vectorizer.train(
            tagged_document,
            total_examples=doc_vectorizer.corpus_count,
            epochs=doc_vectorizer.iter
        )

        # decrease the learning rate
        doc_vectorizer.alpha -= 0.002
        # fix the learning rate, no decay
        doc_vectorizer.min_alpha = doc_vectorizer.alpha

    doc_vectorizer.save('models/doc2vec.model')

    vectorized_tokens = [
        doc_vectorizer.infer_vector(doc.words)
        for doc in tagged_document
    ]

    return np.asarray(vectorized_tokens)


def vectorize_matrix_with_doc2vec(tokens, labels, model=None):
    words_tags = namedtuple('TaggedDocument', ['words', 'tags'])
    str_labels = list(map(lambda x: 'pos' if x == 1 else 'neg', labels))
    tagged_document = [
        words_tags(word, tag)
        for word, tag in zip(tokens, str_labels)
    ]

    doc_vectorizer = Doc2Vec.load(model) if model \
        else generate_doc2vec_matrix(tokens, labels)

    vectorized_tokens = [
        doc_vectorizer.infer_vector(doc.words)
        for doc in tagged_document
    ]

    return np.asarray(vectorized_tokens)
