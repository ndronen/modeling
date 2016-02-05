import sys
sys.setrecursionlimit(5000)
import json
import h5py

import numpy as np

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop

from modeling.layers import ImmutableEmbedding
from modeling.difference import TemporalDifference
import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights)

class GraphMarshaller(modeling.data.GraphMarshaller):
    def marshal(self, data, target=None):
        return {
            'input': data,
            'output': target
            }

    def unmarshal(self, output):
        return output['output']

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def build_residual_model(args):
    graph = Graph()

    graph.add_input('input', input_shape=(args.input_width,), dtype='int')

    graph.add_node(build_embedding_layer(args), name='embedding', input='input')

    graph.add_node(build_convolutional_layer(args), name='conv', input='embedding')
    prev_layer = 'conv'
    if args.batch_normalization:
        graph.add_node(BatchNormalization(), name='conv_bn', input=prev_layer)
        prev_layer = 'conv_bn'
    graph.add_node(Activation('relu'), name='conv_relu', input=prev_layer)

    graph.add_node(build_pooling_layer(args), name='pool', input='conv_relu')

    graph.add_node(Flatten(), name='flatten', input='pool')
    prev_layer = 'flatten'

    # Add some number of fully-connected layers without skip connections.
    for i in range(args.n_fully_connected):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(args, n_hidden=args.n_hidden)
        graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if args.batch_normalization:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        if args.dropout_fc_p > 0.:
            graph.add_node(Dropout(args.dropout_fc_p), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'
    
    # Add sequence of residual blocks.
    for i in range(args.n_residual_blocks):
        # Add a fixed number of layers per residual block.
        block_name = '%02d' % i

        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
        prev_layer = block_input_layer = block_name+'input'

        try:
            n_layers_per_residual_block = args.n_layers_per_residual_block
        except AttributeError:
            n_layers_per_residual_block = 2

        for layer_num in range(n_layers_per_residual_block):
            layer_name = 'h%s%02d' % (block_name, layer_num)
    
            l = build_dense_layer(args, n_hidden=args.n_hidden)
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if args.batch_normalization:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < n_layers_per_residual_block:
                a = Activation('relu')
                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
                prev_layer = layer_name+'relu'
                if args.dropout_fc_p > 0.:
                    graph.add_node(Dropout(args.dropout_fc_p), name=layer_name+'do', input=prev_layer)
                    prev_layer = layer_name+'do'

        graph.add_node(Identity(), name=block_name+'output', inputs=[block_input_layer, prev_layer], merge_mode='sum')
        graph.add_node(Activation('relu'), name=block_name+'relu', input=block_name+'output')
        prev_layer = block_input_layer = block_name+'relu'

    graph.add_node(build_dense_layer(args, args.n_classes,
            activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name='output', input='softmax')

    load_weights(args, graph)

    optimizer = build_optimizer(args)

    graph.compile(loss={'output': args.loss}, optimizer=optimizer)

    return graph

def build_ordinary_model(args):
    model = Sequential()
    model.add(build_embedding_layer(args))
    if args.dropout_embedding_p > 0.:
        model.add(Dropout(args.dropout_embedding_p))
    model.add(build_convolutional_layer(args))
    if args.batch_normalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    if args.dropout_conv_p > 0.:
        model.add(Dropout(args.dropout_conv_p))

    model.add(build_pooling_layer(args))
    model.add(Flatten())

    for i in range(args.n_fully_connected):
        model.add(build_dense_layer(args))
        if args.batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if args.dropout_fc_p > 0.:
            model.add(Dropout(args.dropout_fc_p))

    model.add(build_dense_layer(args, args.n_classes,
            activation='softmax'))

    load_weights(args, model)

    optimizer = build_optimizer(args)

    model.compile(loss=args.loss, optimizer=optimizer)

    if args.verbose:
        for k,v in json.loads(model.to_json()).items():
            if k == 'layers':
                for l in v:
                    print('  => %s' % l['name'])

    return model

def build_model(args):
    np.random.seed(args.seed)

    if args.n_residual_blocks > 0:
        return build_residual_model(args)
    else:
        return build_ordinary_model(args)

