import sys
sys.setrecursionlimit(5000)
import json
import h5py

import numpy as np

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop

from modeling.layers import ImmutableEmbedding
from modeling.difference import TemporalDifference
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights)

def error_free_examples(path):
    f = h5py.File(path)
    # Target_code is 0 when the preposition in the example is the original
    # preposition in the corpus and 1 when the preposition has been randomly
    # replaced with another one in the confusion set.
    idx = f['target_code'].value == 0
    f.close()
    return idx 

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

class Transpose(Layer):
    def get_output(self, train)
        return self.get_input(train).T

def build_model(args):
    np.random.seed(args.seed)

    graph = Graph()

    graph.add_input('input', input_shape=(args.input_width,), dtype='int')

    graph.add_node(build_embedding_layer(args), 
            input='input', name='embedding')

    graph.add_node(LSTM(args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True),
        input='embedding', name='lstm0')

    # Build a sequential MLP to add as an attention module.
    attention = Sequential()
    attention.add(Dense(100, activation='relu', input_shape=(args.n_units,))
    attention.add(Dense(100, activation='relu'))
    attention.add(Activation('softmax'))

    graph.add_node(Transpose(), input='lstm0', name='transpose0')
    graph.add_node(attention(), input='transpose', name='attention')
    graph.add_node(Transpose(), input='attention', name='transpose1')

    graph.add_node(inputs=['lstm0', 'transpose1'], merge_mode='mul')

    graph.add_node(build_dense_layer(args, args.n_classes,
            activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name='output', input='softmax')

    load_weights(args, graph)

    optimizer = build_optimizer(args)

    graph.compile(loss={'output': args.loss}, optimizer=optimizer)

    return graph
