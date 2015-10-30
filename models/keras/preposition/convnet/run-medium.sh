#!/bin/bash -xe

N=10000000

#--extra-train-file $(ls data/preposition/prepositions-all-new-train-$N/* | grep -v 00.h5) \

embedding_weights=data/preposition/prepositions-all-new-weights.npy

./train_keras.py \
    models/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    XwindowNULL \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = XwindowNULL, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=500 , n_hidden=1000, n_word_dims=300 (pre-trained, frozen), 3 hidden layers, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=500 n_hidden=1000 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --log

./train_keras.py \
    models/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    XwindowNULL X \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = XwindowNULL X, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=500 , n_hidden=1000, n_word_dims=300 (pre-trained, frozen), 3 hidden layers, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=500 n_hidden=1000 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --log

