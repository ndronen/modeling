#!/bin/bash -xe

./train_keras.py models/keras/spelling/lstm \
    data/spelling/birbeck-train.h5 \
    data/spelling/birbeck-valid.h5 \
    word \
    --target-name is_real_word \
    --n-embeddings 56 \
    --model-cfg n_units=20 n_embed_dims=25 patience=1000 train_embeddings=true embedding_init=uniform optimizer=Adam \
    --shuffle \
    --log \
    --confusion-matrix \
    --classification-report \
    --class-weight-auto \
    --class-weight-exponent 5 \
    --n-epochs 350
