#!/bin/bash -e

model_dir=models/keras/spelling/convnet
data_dir=data/spelling/experimental/
distance=1
errors=3

experiment_name=$(echo $0 | sed -r 's/.*-(exp[0-9][0-9]-[a-zA-Z0-9][a-zA-Z0-9]*).sh/\1/')
experiment_dir=$model_dir/$experiment_name
mkdir -p $experiment_dir

for operation in delete
do
    n_embed_dims=100
    n_filters=1000
    filter_width=5
    n_fully_connected=2
    n_hidden=100

    echo ./train_keras.py $model_dir \
        $data_dir/$operation-${errors}errors1word-distance-$distance.h5 \
        $data_dir/$operation-${errors}errors1word-distance-$distance.h5 \
        word \
        --model-dest $experiment_dir/op_${operation}_n_embed_dims_${n_embed_dims}_n_filters_${n_filters}_filter_width_${filter_width}_n_fully_connected_${n_fully_connected}_n_hidden_${n_hidden} \
        --target-name target \
        --n-embeddings 61 \
        --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=${n_fully_connected} n_hidden=$n_hidden patience=10 \
        --shuffle \
        --confusion-matrix \
        --classification-report \
        --class-weight-auto \
        --class-weight-exponent 3 \
        --early-stopping-metric f2 \
        --n-validation 100000 \
        --log \
        --verbose
done | parallel --gnu -j 2
