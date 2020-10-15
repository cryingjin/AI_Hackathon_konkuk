from sklearn.feature_extraction.text import TfidfVectorizer


def generate_word_dict(document):
    tfidfv = TfidfVectorizer(max_features=3000).fit(document)
    score = {
        key: value for key, value in zip(
            sorted(tfidfv.vocabulary_, key=lambda x: tfidfv.vocabulary_[x]),
            tfidfv.transform(document).toarray().sum(axis=0)
        )
    }
    top_pos = sorted(score, key=lambda x: score[x], reverse=True)[:200]
    return top_pos


"""
Feature Engineering
"""
# 단어사전 만들어서 변수 생성
top_pos = generate_word_dict(pos_doc)
top_neg = sgenerate_word_dict(neg_doc)

pos_only = set(top_pos) - (set(top_pos) & set(top_neg))
neg_only = set(top_neg) - (set(top_pos) & set(top_neg))

pos_cnt = [len(set(i) & pos_only) for i in train_tokens]
neg_cnt = [len(set(i) & neg_only) for i in train_tokens]
ratio = (np.array(pos_cnt)+1)/(np.array(neg_cnt)+1)

X_train_addfeature = np.concatenate(
    (X_train_np, pos_cnt, neg_cnt, ratio), axis=1)
