from data_utils import load_pickle_data
from embedding_utils import (
    vectorize_matrix_with_word2vec,
    vectorize_matrix_with_doc2vec
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier

if __name__ == '__main__':
    # 1. Load data
    # 2. Generate tokens
    # 현재 생성된 token 을 load 하는 방식으로만 통합 되어있음
    base_dir = 'data'
    train_tokens, train_labels, test_tokens, test_labels \
        = load_pickle_data(base_dir)

    # 3. Embedding
    embed_types = ['word2vec', 'doc2vec']
    train_x_dict = {
        'count': None,
        'word2vec': vectorize_matrix_with_word2vec(train_tokens),
        'doc2vec': vectorize_matrix_with_doc2vec(
            train_tokens,
            train_labels,
            model='model/doc2vec.model'
        ),
    }
    test_x_dict = {
        'count': None,
        'word2vec': vectorize_matrix_with_word2vec(test_tokens),
        'doc2vec': vectorize_matrix_with_doc2vec(
            test_tokens, test_labels,
            model='model/doc2vec.model'
        ),
    }

    # 5. Train classifiers with those features
    total_classifiers = []

    for embed_type in embed_types:
        train_x = train_x_dict[embed_type]
        test_x = test_x_dict[embed_type]

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(train_x, train_labels)
        nb_predictions = nb.predict(test_x)
        nb_acc_score = accuracy_score(test_labels, nb_predictions)
        total_classifiers.append(('{}-NB'.format(embed_type), nb))

        # SVM with BaggingClassifier
        estimator = LinearSVC()
        bag = BaggingClassifier(
            base_estimator=estimator,
            n_estimators=10,
            max_samples=(1 / 10),
            max_features=1,
            n_jobs=-1
        )
        bag.fit(train_x, train_labels)
        bag_predictions = bag.predict(test_x)
        bag_acc_score = accuracy_score(test_labels, bag_predictions)
        total_classifiers.append(('{}-BAG'.format(embed_type), bag))

        # Print result
        print('{} | NB: {} | BAG: {}'.format(
            embed_type, nb_acc_score, svm_acc_score))
