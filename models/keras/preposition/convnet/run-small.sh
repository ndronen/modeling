#!/bin/bash -xe

N=1000000

embedding_weights=data/preposition/prepositions-all-new-weights.npy

function train() {
    n_filters=$1
    shift
    filter_width=$1
    shift
    features=$@

    features_name=$(echo $features | sed 's, ,-,g')
    dest=$features_name-$n_filters-$filter_width

    ./train_keras.py \
        models/keras/preposition/convnet \
        data/preposition/prepositions-all-new-train-$N-balanced.h5 \
        data/preposition/prepositions-all-new-validate-balanced.h5 \
        $features \
        --model-dest models/keras/preposition/convnet/small/feature-evaluation/$dest \
        --target-name original_word_code \
        --target-data data/preposition/prepositions-all-new-target-data.json \
        --description "comparing inputs with convnets - input = $features, target = original_word_code, contrasting, $N training examples, Adagrad, patience=5, n_filters=$n_filters, filter_width=$filter_width, n_word_dims=300 (pre-trained, frozen), 1 hidden layer, shuffled data" \
        --n-vocab 83064 \
        --model-cfg optimizer=Adagrad regularization_layer="dropout" n_filters=$n_filters n_word_dims=300 embedding_weights=$embedding_weights train_embeddings=false filter_width=$filter_width patience=5 \
        --n-validation 20000 \
        --n-epochs 10 \
        --shuffle \
        --log
}

function xval5() {
    features=$@
    for filter_width in 2 3 5
    do
        for n_filters in 100 
        do
            train $n_filters $filter_width $features
        done
    done
}

function xval7() {
    features=$@
    for filter_width in 2 3 5 7
    do
        for n_filters in 100 
        do
            train $n_filters $filter_width $features
        done
    done
}

function xval9() {
    features=$@
    for filter_width in 2 3 5 7 9
    do
        for n_filters in 100 
        do
            train $n_filters $filter_width $features
        done
    done
}

xval5 Xwindow
xval7 Xwindow7
xval9 Xwindow9

xval5 XwindowNULL X
xval7 Xwindow7NULL X
xval9 Xwindow9NULL X

xval9 X

xval5 XwindowNULL
xval7 Xwindow7NULL
xval9 Xwindow9NULL

xval5 Xwindow X
xval7 Xwindow7 X
xval9 Xwindow9 X
