import joblib
import numpy as np
import pickle
from gensim.models import Word2Vec, Doc2Vec
from data_utils import load_text_data, tokenize
from embedding_utils import (
    vectorize_matrix_with_word2vec,
    vectorize_matrix_with_doc2vec
)
from features import mk_feature


if __name__ == "__main__":
    # load and preprocess
    indices, documents = load_text_data('data/rating_eval.txt')
    document_tokens = [tokenize(document) for document in documents]

    # load vectorizer and embed
    # count vectorizer
    count_v = joblib.load('model/count_vectorizer.pickle')
    count_v_test = count_v.transform(
        [' '.join(tokens) for tokens in document_tokens]).toarray()

    additional_feature = mk_feature(documents, document_tokens)

    # word2vec
    w2v_model = Word2Vec.load('model/model_w2v_128.model')
    w2v_test = np.concatenate(
        (
            vectorize_matrix_with_word2vec(
                document_tokens,
                model=w2v_model
            ),
            mk_feature(documents, document_tokens)
        ),
        axis=1
    )

    # doc2vec
    d2v_model = Doc2Vec.load('model/doc2vec.model')
    d2v_test = np.concatenate(
        (
            vectorize_matrix_with_doc2vec(document_tokens, model=d2v_model),
            mk_feature(documents, document_tokens)
        ),
        axis=1
    )

    # load model and predict
    raw_predictions = []

    # CountVectorizer
    count_v_model = joblib.load('model/model-mg.pickle')
    raw_predictions.append(count_v_model.predict_proba(count_v_test))

    # word2vec + KNN
    w2v_knn_model = joblib.load('model/model_bag_knn.pickle')
    raw_predictions.append(w2v_knn_model.predict_proba(w2v_test))

    # word2vec + SVM
    w2v_svm_model = joblib.load('model/model_bag_svm.pickle')
    raw_predictions.append(w2v_svm_model.predict_proba(w2v_test))

    # doc2vec + SVM
    d2v_svm_model = joblib.load('model/doc2vec_svm_model.pickle')
    raw_predictions.append(d2v_svm_model.predict_proba(d2v_test))

    # ensemble
    predictions = np.array(raw_predictions).mean(axis=0)
    final_predictions = [0 if a > b else 1 for (a, b) in predictions]

    # submit final result
    with open('1팀.pred', 'w', encoding='utf-8') as fp:
        for index, prediction in zip(indices, final_predictions):
            fp.write('{}\t{}\n'.format(index, prediction))
