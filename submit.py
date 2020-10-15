import joblib
import numpy as np
from gensim.models import Word2Vec
from data_utils import load_raw_data, tokenize
from embedding_utils import vectorize_matrix_with_word2vec

if __name__ == "__main__":
    # load and preprocess
    indices, documents = load_raw_data('data/submit-data.txt')
    document_tokens = [tokenize(document) for document in documents]

    # load vectorizer and embed
    # count vectorizer
    count_v = joblib.load('data/count_vectorizer.pickle')
    count_v_test = count_v.transform(
        [' '.join(tokens) for tokens in document_tokens]).toarray()

    # word2vec
    w2v_model = Word2Vec.load('model/model_w2v_128.model')
    w2v_test = vectorize_matrix_with_word2vec(
        document_tokens,
        model=w2v_model
    )

    # dot2vec

    # load model and predict
    raw_predictions = []

    count_v_model = joblib.load('data/model-mg.pickle')
    raw_predictions.append(count_v_model.predict(count_v_test))

    w2v_knn_model = joblib.load('data/model_bag_knn.pickle')
    raw_predictions.append(w2v_knn_model.predict(w2v_test))

    w2v_svm_model = joblib.load('data/model_bag_svm.pickle')
    raw_predictions.append(w2v_svm_model.predict(w2v_test))

    # ensemble
    ensembled_predictions = np.array(raw_predictions).mean(axis=0)
    final_predictions = [
        1 if pred >= 0.5 else 0
        for pred in ensembled_predictions
    ]

    # submit final result
    with open('1팀.pred', 'w', encoding='utf-8') as fp:
        for index, prediction in zip(indices, final_predictions):
            fp.write('{}\t{}\n'.format(index, prediction))
