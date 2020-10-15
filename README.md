# tfidf, kai, tfidf+kai/ one-hot, count 

train_x_size          1500    15000   15000   

train_y_size          500     5000    5000

tfidf_dict_size       1000    1000    5000

kai_dict_size         1000    1000    5000

tfidf_sprs(onehot)    0.766   0.8266  0.8266

kai_sprs(onehot)      0.758   0.823   0.8254

tfidf+kai_sprs(onehot)   0.758   0.8224   0.826

tfidf_cnt(count)      0.764   0.8238  0.8238

kai_cnt(count)        0.76    0.8212  0.8242

tfidf+kai_cnt(count)  0.764   0.822   0.8228

train_x_size 150000이상 -> 메모리 터짐

(벡터 만들 때 본인이 직접 for문으로 만들어서 그런듯. 희소행렬(sparse matrix)로 만들어도 터짐. CountVectorize 써야됢.....)

tfidf or kai 단어사전 구축하고, 그 단어사전의 단어를 포함하고 있지 않은 문장 train_x에서 제외시키는 방법으로도 결과는 거의 동일 했다.
