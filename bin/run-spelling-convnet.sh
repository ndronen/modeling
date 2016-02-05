#!/bin/bash -e

model_dir=models/keras/spelling/convnet
data_dir=data/spelling/experimental/
distance=1
errors=3
nonce_interval=-nonce-interval-3

mkdir -p $model_dir/crossval

#for operation in delete insert substitute transpose
#for nonce in "" "-nonce-interval-3"
#do
for operation in delete
do
    for n_embed_dims in 100 
    do
        for n_filters in 1000 
        do
            for filter_width in 5
            do
                for n_hidden in 100
                do
                    for n_fully_connected in 1 2 3 4
                    do
                        echo ./train_keras.py $model_dir \
                            $data_dir/$operation-${errors}errors1word-distance-$distance${nonce}.h5 \
                            $data_dir/$operation-${errors}errors1word-distance-$distance${nonce}.h5 \
                            word \
                            --model-dest $model_dir/crossval/op_${operation}_n_embed_dims_${n_embed_dims}_n_filters_${n_filters}_filter_width_${filter_width}_n_fully_connected_${n_fully_connected}_n_hidden_${n_hidden} \
                            --target-name target \
                            --n-embeddings 61 \
                            --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_hidden=$n_hidden n_fully_connected=${n_fully_connected} patience=3 \
                            --shuffle \
                            --confusion-matrix \
                            --classification-report \
                            --class-weight-auto \
                            --class-weight-exponent 3 \
                            --early-stopping-metric f2 \
                            --n-validation 100000 \
                            --log
                    done
                done
            done
        done
    done
done | parallel --gnu -j 2
