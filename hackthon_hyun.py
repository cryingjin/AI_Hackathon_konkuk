# ready
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
import warnings
warnings.filterwarnings(action='ignore')
import pickle
# visualization
from matplotlib import pyplot as plt
plt.style.use('seaborn')
%matplotlib inline
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
# Font
import matplotlib.font_manager as fm
fm._rebuild()
plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

from sklearn.metrics import accuracy_score
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import gensim
from gensim import corpora
import pickle
import random
from konlpy.tag import Okt
import nltk
from gensim.models import Doc2Vec
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier


## utils
def load_pickle(path):
    with open(path,'rb') as fr:
        return pickle.load(fr)

def save_pickle(path,file):
    with open(path,'wb') as f:
        pickle.dump(file,f)



## load data
train_data = pd.read_csv('ratings_train.txt', header = 0, delimiter = '\t', quoting = 3)
test_data = pd.read_csv('ratings_test.txt', header = 0, delimiter = '\t', quoting = 3)
train_tokens = load_pickle('train_tokens.pickle')
train_label = load_pickle('train_label.pickle')


## 전처리
train_data.drop_duplicates(subset = ['document'], inplace=True)
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['document'].replace('', np.nan, inplace=True)
train_data = train_data.dropna(how='any')
train_data

test_data.drop_duplicates(subset = ['document'], inplace=True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')
test_data


## 불용어 처리 및 tokenizing
def tokenizing(data):
    stopwords = ['의','가','이','은','들','을','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다','에서','되','그','수','나','것','하','있','보','주','아니','등','같','때','년','가','한','지','오','말','일','다','이다']

    okt=Okt()  
    def tknz(sentence):
        s = okt.pos(sentence,norm=True,stem=True)
        x = []
        for w in s:
            if w[1] == 'Adjective' or w[1] == 'Adverb' or w[1]=='Noun':
                x.append(w[0])
            else: 
                continue
        return x

    tokens = []
    for s in data:
        x = tknz(str(s))
        tem = [word for word in x if not word in stopwords] 
        tokens.append(tem)

    return tokens



## 토큰 긍/부정 분리(feature engineeing에서 변수 만드는데 사용)
train_label = pd.DataFrame(train_label)

neg_train_idx = train_label[train_label['label']==0].index
pos_train_idx = train_label[train_label['label']==1].index

neg_train = []
for i in neg_train_idx:
    neg_train.append(train_tokens[i])

pos_train = []
for i in pos_train_idx:
    pos_train.append(train_tokens[i])

def mk_join_doc(corpus):
    doc = []
    for w in corpus:
        tem = " ".join(w)
        doc.append(tem)
    return doc

pos_doc = mk_join_doc(pos_train)
neg_doc = mk_join_doc(neg_train)


"""
Doc2vec
"""
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
TaggedDocument = namedtuple('TaggedDocument', 'words tags')

# 0,1 문자열로 mapping
def mapping(x):
    if x==0:
        return 'neg'
    else:
        return 'pos'

train_label['label'] = train_label['label'].map(mapping)


tagged_train_docs = [TaggedDocument(d, c) for d, c in zip(train_tokens,train_label['label'])]
# tagged_test_docs = [TaggedDocument(d, c) for d, c in zip(X_test,y_test)]

# cpu 병렬 처리
import multiprocessing
cores = multiprocessing.cpu_count()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

doc_vectorizer = Doc2Vec(
    dm=0,            # PV-DBOW / default 1
    dbow_words=1,    # w2v simultaneous with DBOW d2v / default 0
    window=5,        # distance between the predicted word and context words
    size=128,        # vector size
    alpha=0.025,     # learning-rate
    seed=1234,
    min_count=1,     # ignore with freq lower
    min_alpha=0.025, # min learning-rate
    workers=cores,   # multi cpu
    hs = 1,          # hierarchical softmax / default 0
    negative = 5,   # negative sampling / default 5
    epochs = 20
)

# doc2vec에 토큰 load
doc_vectorizer.build_vocab(tagged_train_docs)

# train
for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
    doc_vectorizer.alpha -= 0.002 # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay

# 학습 doc2vec 저장
model_name = 'Doc2vec_128_145791.model'
doc_vectorizer.save(model_name)

# doc2vec load
model_name = 'Doc2vec(dbow+w,d300,n10,hs,w8,mc20,s0.001,t24).model'
doc_vectorizer = Doc2Vec.load(model_name)

# mk modeling set
X_train = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
y_train = [doc.tags for doc in tagged_train_docs]
X_test = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
y_test = [doc.tags for doc in tagged_test_docs]

y_train_np = np.asarray([0 if c == 'neg' else 1 for c in y_train], dtype=int)
y_test_np = np.asarray([0 if c == 'neg' else 1 for c in y_test], dtype=int)

X_train_np = np.asarray(X_train)
X_test_np = np.array(X_test)


"""
Feature Engineering
"""
## 단어사전 만들어서 변수 생성
tfidfv = TfidfVectorizer(max_features=3000).fit(pos_doc)
score = {key : value for key, value in zip(sorted(tfidfv.vocabulary_, key = lambda x : tfidfv.vocabulary_[x]), tfidfv.transform(pos_doc).toarray().sum(axis=0))}
top_pos = sorted(score, key = lambda x : score[x],reverse=True)[:200]
print(top_pos)

tfidfv = TfidfVectorizer(max_features=3000).fit(neg_doc)
score = {key : value for key, value in zip(sorted(tfidfv.vocabulary_, key = lambda x : tfidfv.vocabulary_[x]), tfidfv.transform(neg_doc).toarray().sum(axis=0))}
top_neg = sorted(score, key = lambda x : score[x],reverse=True)[:200]
print(top_neg)

pos_only = set(top_pos) - (set(top_pos) & set(top_neg))
neg_only = set(top_neg) - (set(top_pos) & set(top_neg))

pos_cnt = [len(set(i)&pos_only) for i in train_tokens]    
neg_cnt = [len(set(i)&neg_only) for i in train_tokens]    
ratio = (np.array(pos_cnt)+1)/(np.array(neg_cnt)+1)

X_train_addfeature = np.concatenate((X_train_np,pos_cnt,neg_cnt,ratio),axis=1)

X_train_addfeature.shape


"""
Modeling
"""
X_train,X_test,y_train,y_test = train_test_split(X_train_np,y_train_np)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
predictions_ = classifier.predict(X_test).tolist()
print('Accuracy: %.10f' % accuracy_score(y_test, predictions_))

estimator = SVC()
n_estimators = 10
n_jobs = -1
model = BaggingClassifier(base_estimator=estimator,
                          n_estimators=n_estimators,
                          max_samples=1./n_estimators,max_features=1.0,
                          n_jobs=n_jobs)

model.fit(X_train[:10000],y_train[:10000])
accuracy_score(model.predict(X_test),y_test)