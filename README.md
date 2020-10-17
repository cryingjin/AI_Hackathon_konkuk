# AI_Hackathon_konkuk

## Usage

Clone

```
$ git clone https://github.com/HyunLee103/AI_Hackathon_konkuk/
```

Add data and model files so directory should look like:

```
.
├── README.md
├── data
│   ├── test_label.pickle
│   ├── test_tokens.pickle
│   ├── train_label.pickle
│   └── train_tokens.pickle
├── data_utils.py
├── embedding_utils.py
├── features.py
├── main.py
└── model
    ├── doc2vec.model
    └── word2vec.model
```

then:

```
$ python main.py
```

will print final results for all classifiers used.

### Submit

for submission, run:

```
$ python submit.py
```
