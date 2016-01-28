import sys
#sys.setrecursionlimit(5000)
import json
import h5py
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights)

def build_model(args):
    np.random.seed(args.seed)

    model = Sequential()
    model.add(build_embedding_layer(args))
    if args.dropout_embedding_p > 0.:
        model.add(Dropout(args.dropout_embedding_p))
    model.add(build_convolutional_layer(args))
    if 'normalization' in args.regularization_layer:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if args.dropout_conv_p > 0.:
        model.add(Dropout(args.dropout_conv_p))

    model.add(build_pooling_layer(args))
    model.add(Flatten())

    for i in range(args.n_fully_connected):
        model.add(build_dense_layer(args))
        if 'normalization' in args.regularization_layer:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if 'dropout' in args.regularization_layer:
            model.add(Dropout(args.dropout_p))

    model.add(build_dense_layer(args, args.n_classes,
            activation='softmax'))

    load_weights(args, model)

    optimizer = build_optimizer(args)

    model.compile(loss=args.loss, optimizer=optimizer)

    for k,v in json.loads(model.to_json()).items():
        if k == 'layers':
            for l in v:
                print('  %s => %s' %(l['name'], l))

    return model
