# coding: utf-8

import numpy as np
import pandas as pd
import modeling.utils
import spelling.baseline

def mark(words):
    return ['^'+w+'$' for w in words]

def build_index():
    train_hdf5_file = 'data/spelling/experimental/op-transpose-distance-1-errors-per-word-3.h5'
    train_h5 = h5py.File(train_hdf5_file)

    train_csv_file = 'data/spelling/experimental/op-transpose-distance-1-errors-per-word-3.csv'
    train_df = pd.read_csv(train_csv_file, sep='\t', encoding='utf8')
    words = train_df.real_word.tolist()
    marked_words = mark(words)

    X_train = train_h5['marked_chars'].value
    index_size = np.max(X_train)
    i = 0
    index = {}

    while len(index) < index_size:
        marked_word = marked_words[i]
        row = X_train[i]

        for j,idx in enumerate(row):
            if idx == 0:
                break
            index[marked_word[j]] = idx

        i += 1

    return index

index = build_index()

model_dir = 'models/keras/spelling/convnet/exp03-inputs/op_transpose_n_ops_1_n_errors_per_word_3'

df = pd.read_csv('../spelling/data/aspell-dict.csv.gz', sep='\t', encoding='utf8')
words = df.word.tolist()
vocab = set(words)

lm = spelling.baseline.CharacterLanguageModel('witten-bell', order=3)
lm.fit(words)

model, model_cfg = modeling.utils.load_model(model_dir, model_weights=True)

bins = np.arange(0, 1, .1)
outputs = {}
histograms = {}

for order in range(1, 4):
    print('order %d' % order)
    generated = []
    # Generate 500k words, controlling for length and excluding those
    # that are already in the vocabulary.  Only keep the first 100k
    # of those that satisfy our requirements.
    for g in lm.generate(order, 500000):
        if len(g) < 5 or len(g) > 10:
            continue
        if g in vocab:
            continue
        generated.append(g)
        if len(generated) == 100000:
            break

    marked = mark(generated)
    X = np.zeros((len(marked), input_width))
    for i,word in enumerate(marked):
        for j,chr in enumerate(word):
            X[i,j] = index[chr]
        
    output = zip(generated, model.predict(X)[:, 1])
    outputs[order] = output
    histograms[order] = np.histogram([o[1] for o in output], bins=bins)
