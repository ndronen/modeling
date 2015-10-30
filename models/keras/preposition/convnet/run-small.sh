#!/bin/bash -xe

N=1000000

embedding_weights=data/preposition/prepositions-all-new-weights.npy

./train_keras.py \
    models/keras/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    X \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = X, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=100 , n_hidden=200, n_word_dims=300 (pre-trained, frozen), 1 hidden layers, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=100 n_hidden=200 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --no-save
    #--log

exit

./train_keras.py \
    models/keras/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    Xwindow X \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = Xwindow X, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=100 , n_hidden=200, n_word_dims=300 (pre-trained, frozen), 1 hidden layer, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=100 n_hidden=200 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --log

./train_keras.py \
    models/keras/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    Xwindow \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = Xwindow, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=100 , n_hidden=200, n_word_dims=300 (pre-trained, frozen), 1 hidden layer, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=100 n_hidden=200 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --log

./train_keras.py \
    models/keras/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    XwindowNULL X \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = XwindowNULL X, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=100 , n_hidden=200, n_word_dims=300 (pre-trained, frozen), 1 hidden layer, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=100 n_hidden=200 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --log

./train_keras.py \
    models/keras/preposition/convnet \
    data/preposition/prepositions-all-new-train-$N.h5 \
    data/preposition/prepositions-all-new-validate.h5 \
    XwindowNULL \
    --target-name original_word_code \
    --target-data data/preposition/prepositions-all-new-target-data.json \
    --description "comparing inputs with convnets - input = XwindowNULL, target = original_word_code, contrasting, $N training examples, Adagrad, n_filters=100 , n_hidden=200, n_word_dims=300 (pre-trained, frozen), 1 hidden layer, shuffled data" \
    --n-vocab 83064 \
    --model-cfg optimizer=Adagrad regularization_layer="" patience=10 n_filters=100 n_hidden=200 n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false \
    --n-validation 20000 \
    --classification-report \
    --shuffle \
    --n-epochs 10 \
    --log
