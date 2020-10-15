from data_utils import load_data
from embedding_utils import vectorize_matrix_with_word2vec, generate_doc2vec_matrix


if __name__ == '__main__':
    # 1. Load data
    # 2. Generate tokens
    # 현재 생성된 token 을 load 하는 방식으로만 통합 되어있음
    base_dir = 'data'
    train_tokens, train_labels, test_tokens, test_labels = load_data(base_dir)

    # 3. Embedding
    embed_types = ['count', 'word2vec', 'doc2vec']
    train_x_dict = {
        'count': None,
        'word2vec': vectorize_matrix_with_word2vec(train_tokens),
        'doc2vec': generate_doc2vec_matrix(train_tokens, train_labels),
    }
    test_x_dict = {
        'count': None,
        'word2vec': vectorize_matrix_with_word2vec(test_tokens),
        'doc2vec': generate_doc2vec_matrix(test_tokens, test_labels),
    }

    # TODO: 4. Feature Engineering

    # 5. Train classifiers with those features
    for embed_type in embed_types:
        train_x = train_x_dict[embed_type]
        test_x = test_x_dict[embed_type]

        # Naive Bayes
        nb = GaussianNB()
        nb.fit(train_x, train_labels)
        nb_predictions = nb.predict(test_x, test_labels)
        nb_acc_score = accuracy_score(test_labels, nb_predictions)

        # SVM with BaggingClassifier
        estimator = SVC()
        svm = BaggingClassifier(
            base_estimator=10,
            n_estimators=-1,
            max_samples=(1 / 10),
            max_features=1,
            n_jobs=-1
        )
        svm.fit(train_x, train_labels)
        svm_predictions = svm.predict(test_x, test_labels)
        svm_acc_score = accuracy_score(test_labels, svm_predictions)

        # Print result
        print('{} | NB: {} | SVM: {}'.format(
            embed_type, nb_acc_score, svm_acc_score))

    # TODO: 6. Ensemble
