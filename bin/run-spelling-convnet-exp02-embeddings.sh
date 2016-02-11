#!/bin/bash -e

model_dir=models/keras/spelling/convnet
data_dir=data/spelling/experimental/
distance=1
errors=3

experiment_name=$(echo $0 | sed -r 's/.*-(exp[0-9][0-9]-..*).sh/\1/')
experiment_dir=$model_dir/$experiment_name
mkdir -p $experiment_dir

operation=delete
n_embed_dims=56
n_filters=10
filter_width=6
n_fully_connected=0
n_hidden=0

for embedding_init in identity orthogonal uniform normal 
do
    for train_embeddings in false true
    do
        model_dest=$experiment_dir/op_${operation}_n_embed_dims_${n_embed_dims}_n_filters_${n_filters}_filter_width_${filter_width}_n_fully_connected_${n_fully_connected}_n_hidden_${n_hidden}_embedding_init_${embedding_init}_train_embeddings_${train_embeddings}
            #--model-dest $model_dest \
        echo $model_dest
        ./train_keras.py $model_dir \
            $data_dir/op-${operation}-distance-${distance}-errors-per-word-${errors}.h5 \
            $data_dir/op-${operation}-distance-${distance}-errors-per-word-${errors}.h5 \
            chars \
            --target-name binary_target \
            --n-embeddings 56 \
            --model-cfg n_embed_dims=$n_embed_dims n_filters=$n_filters filter_width=$filter_width n_fully_connected=${n_fully_connected} n_hidden=$n_hidden embedding_init=$embedding_init train_embeddings=$train_embeddings optimizer=SGD learning_rate=0.001 momentum=0.0 decay=0.0 \
            --shuffle \
            --confusion-matrix \
            --classification-report \
            --class-weight-auto \
            --class-weight-exponent 3 \
            --verbose \
            --n-train 50000 \
            --n-epochs 3 \
            --no-save
            #--log \
    done 
done
#| parallel --gnu -j 2
