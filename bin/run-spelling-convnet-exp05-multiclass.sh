#!/bin/bash -ex

model_dir=models/keras/spelling/convnet
data_dir=data/spelling/experimental/
distance=1
errors=3

experiment_name=$(echo $0 | sed -r 's/.*-(exp[0-9][0-9]-..*).sh/\1/')
experiment_dir=$model_dir/$experiment_name
mkdir -p $experiment_dir

for operation in delete
do
    for n_embed_dims in 100
    do
        for n_filters in 300 
        do
            for filter_width in 8
            do
                for n_fully_connected in 2
                do
                    for n_residual_blocks in 1
                    do
                        for n_hidden in 300
                        do
                            ./train_keras.py $model_dir \
                                $data_dir/op-$operation-distance-$distance-errors-per-word-${errors}/000.h5 \
                                $data_dir/op-$operation-distance-$distance-errors-per-word-${errors}/000.h5 \
                                chars \
                                --target-name multiclass_target \
                                --model-dest $experiment_dir/op_${operation}_n_embed_dims_${n_embed_dims}_n_filters_${n_filters}_filter_width_${filter_width}_n_fully_connected_${n_fully_connected}_n_residual_blocks_${n_residual_blocks}_n_hidden_${n_hidden}_n_hsm_classes_5000 \
                                --n-embeddings 61 \
                                --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=${n_fully_connected} n_residual_blocks=$n_residual_blocks n_hidden=$n_hidden patience=240 n_hsm_classes=5000 batch_size=8 \
                                --shuffle \
                                --class-weight-auto \
                                --class-weight-exponent 3 \
                                --early-stopping-metric f1 \
                                --verbose \
                                --target-data $data_dir/op-$operation-distance-$distance-errors-per-word-${errors}.json \
                                --extra-train-file $(ls $data_dir/op-$operation-distance-$distance-errors-per-word-${errors}/* | egrep -v '000.h5') \
                                --n-classes 119773 \
                                --log
                        done
                    done
                done
            done
        done
    done
done 
