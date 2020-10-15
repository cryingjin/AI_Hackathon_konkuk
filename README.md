# tfidf, kai, tfidf+kai/ one-hot, count  \n
train_x_size          1500    15000   15000   150000이상 -> 메모리 터짐\n
train_y_size          500     5000    5000\n
tfidf_dict_size       1000    1000    5000\n
kai_dict_size         1000    1000    5000\n
tfidf_sprs(onehot)    0.766   0.8266  0.8266\n
kai_sprs(onehot)      0.758   0.823   0.8254\n
tfidf+kai_sprs(onehot)   0.758   0.8224   0.826\n
tfidf_cnt(count)      0.764   0.8238  0.8238\n
kai_cnt(count)        0.76    0.8212  0.8242\n
tfidf+kai_cnt(count)  0.764   0.822   0.8228\n
