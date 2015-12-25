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

def build_residual_block(name, input_shape, n_hidden, n_skip=2):
    """
    Rough sketch of building blocks of layers for residual learning.
    See http://arxiv.org/abs/1512.03385 for motivation.
    """
    block = Graph()
    input_name = 'x'
    block.add_input(input_name, input_shape=input_shape)

    # The current keras graph implementation doesn't allow you to connect
    # an input node to an output node.  Use Identity to work around that.
    block.add_node(Identity(), name=name+'identity', input=input_name)
    prev_layer = name+'identity'

    for i in range(n_skip):
        layer_name = 'h' + str(i)

        l = build_dense_layer(args, n_hidden=n_hidden)
        block.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name

        # Haven't gotten this to work yet.
        #bn = BatchNormalization()
        #block.add_node(bn, name=layer_name+'bn', input=prev_layer)
        #prev_layer = layer_name+'bn'

        if i < n_skip:
            a = Activation('relu')
            block.add_node(a, name=layer_name+'relu', input=prev_layer)
            prev_layer = layer_name+'relu'

    block.add_output(name=name+'output', inputs=[name+'identity', prev_layer], merge_mode='sum')

    return block

def build_residual_model(args):
    graph = Graph()

    graph.add_input('input', input_shape=(args.input_width,), dtype='int')

    graph.add_node(build_embedding_layer(args), name='embedding', input='input')

    graph.add_node(build_convolutional_layer(args), name='conv', input='embedding')
    prev_layer = 'conv'
    if 'normalization' in args.regularization_layer:
        graph.add_node(BatchNormalization(), name='conv_bn', input=prev_layer)
        prev_layer = 'conv_bn'
    graph.add_node(Activation('relu'), name='conv_relu', input=prev_layer)

    graph.add_node(build_pooling_layer(args), name='pool', input='conv_relu')

    graph.add_node(Flatten(), name='flatten', input='pool')
    prev_layer = 'flatten'

    # Add two dense layers.
    for i in range(2):
        layer_name = 'dense%02d' %i
        l = build_dense_layer(args, n_hidden=args.n_filters)
        graph.add_node(l, name=layer_name, input=prev_layer)
        prev_layer = layer_name
        if 'normalization' in args.regularization_layer:
            graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
            prev_layer = layer_name+'bn'
        if 'dropout' in args.regularization_layer:
            graph.add_node(Dropout(args.dropout_p), name=layer_name+'do', input=prev_layer)
            prev_layer = layer_name+'do'
    
    # Add sequence of residual blocks.
    for i in range(args.n_residual_blocks):
        # Add a fixed number of layers per residual block.
        block_name = '%02d' % i

        graph.add_node(Identity(), name=block_name+'input', input=prev_layer)
        prev_layer = block_input_layer = block_name+'input'

        for layer_num in range(args.n_layers_per_residual_block):
            layer_name = 'h%s%02d' % (block_name, layer_num)
    
            l = build_dense_layer(args, n_hidden=args.n_filters)
            graph.add_node(l, name=layer_name, input=prev_layer)
            prev_layer = layer_name
    
            if 'normalization' in args.regularization_layer:
                graph.add_node(BatchNormalization(), name=layer_name+'bn', input=prev_layer)
                prev_layer = layer_name+'bn'
    
            if i < args.n_layers_per_residual_block:
                a = Activation('relu')
                graph.add_node(Activation('relu'), name=layer_name+'relu', input=prev_layer)
                prev_layer = layer_name+'relu'
                if 'dropout' in args.regularization_layer:
                    graph.add_node(Dropout(args.dropout_p), name=layer_name+'do', input=prev_layer)
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
    #if args.dropout_embedding_p > 0.:
    #    model.add(Dropout(args.dropout_embedding_p))
    model.add(build_convolutional_layer(args))
    if 'normalization' in args.regularization_layer:
        model.add(BatchNormalization())
    model.add(Activation('relu'))

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
        print(k)
        if k == 'layers':
            for l in v:
                print('%s => %s' %(l['name'], l))
        else:
            print(v)

    return model

def build_model(args):
    np.random.seed(args.seed)

    if isinstance(args.n_residual_blocks, int):
        return build_residual_model(args)
    else:
        return build_ordinary_model(args)

