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

class EncartaExamplesWithOKWindows():
    def __init__(self, seed=17):
        self.random_state = np.random.RandomState(seed=seed)
        self.prepositions = set([7, 8, 10, 12, 13, 17, 18, 19, 27])

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

    def transform(self, X, y=None):
        # Select the examples where the middle column is in our
        # preposition set.
        middle_column = X[:, X.shape[1]/2]
        ok = np.array([True] * len(X))
        for i,val in enumerate(middle_column):
            if val not in self.prepositions:
                ok[i] = False
        print('in %d out %d' % (len(X), len(X[ok])))
        if y is not None:
            return X[ok], y[ok]
        else:
            return X[ok]

class TrainingSetRealExamples():
    def __init__(self, seed=17):
        self.random_state = np.random.RandomState(seed=seed)

    def fit_transform(self, X, y=None):
        evens = [i*2 for i in np.arange(X.shape[0]/2)]
        if y is not None:
            return X[evens], y[evens]
        else:
            return X[evens]

    def transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

class RandomPermuter(object):
    def __init__(self, seed=17):
        self.random_state = np.random.RandomState(seed=seed)

    def fit(self, X, y=None):
        pass

    def _transform(self, X, y=None):
        X = X.copy()
        middle_column_idx = np.int(X.shape[1]/2)
        middle_column_values = X[:, middle_column_idx]
        random_values = self.random_state.permutation(middle_column_values)
        X[:, middle_column_idx] = random_values
        if y is None:
            return X
        else:
            return X, y

class ValidationSetRealExamples(RandomPermuter):
    def __init__(self, seed=17):
        self.random_state = np.random.RandomState(seed=seed)

    def fit_transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

    def transform(self, X, y=None):
        evens = [i*2 for i in np.arange(X.shape[0]/2)]
        if y is not None:
            return X[evens], y[evens]
        else:
            return X[evens]

class TrainingSetPrepositionRandomPermuter(RandomPermuter):
    def fit_transform(self, X, y=None):
        return self._transform(X, y)

    def transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

class ValidationSetPrepositionRandomPermuter(RandomPermuter):
    def fit_transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

    def transform(self, X, y=None):
        return self._transform(X, y)

class RandomRegularizer(object):
    def __init__(self, seed=17):
        self.random_state = np.random.RandomState(seed=seed)

    def fit(self, X, y=None):
        pass

    def _transform(self, X, y=None):
        X = X.copy()
        middle_column_idx = np.int(X.shape[1]/2)
        middle_column_values = X[:, middle_column_idx]
        value_set = list(set(middle_column_values.tolist()))
        random_values = []
        for i in np.arange(len(X)):
            current_value = middle_column_values[i]
            while True:
                random_value = self.random_state.choice(value_set)
                if random_value != current_value:
                    random_values.append(random_value)
                    break
        X[:, middle_column_idx] = random_values
        if y is None:
            return X
        else:
            return X, y

class TrainingSetPrepositionRandomRegularizer(RandomRegularizer):
    """
    Takes examples in the form of a vector of indices.  Replaces each
    middle value in each vector with a value from some other example.
    """
    def fit_transform(self, X, y=None):
        return self._transform(X, y)

    def transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

class ValidationSetPrepositionRandomRegularizer(RandomRegularizer):
    def fit_transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

    def transform(self, X, y=None):
        return self._transform(X, y)

class UnconstrainedTrainingSetPrepositionPermuter(object):
    def __init__(self, seed=17):
        self.random_state = np.random.RandomState(seed=seed)

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        X = X.copy()
        middle_column_idx = np.int(X.shape[1]/2)
        middle_column_values = X[:, middle_column_idx]
        random_values = self.random_state.permutation(middle_column_values)
        X[:, middle_column_idx] = random_values
        if y is None:
            return X
        else:
            return X, y

    def transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y


def real_examples(path):
    f = h5py.File(path)
    # Target_code is 0 when the preposition in the example is the original
    # preposition in the corpus and 1 when the preposition has been randomly
    # replaced with another one in the confusion set.
    idx = f['target_code'].value == 0
    f.close()
    return idx 

def random_regularization_examples(path):
    f = h5py.File(path)
    idx = f['target_code'].value == 1
    f.close()
    return idx 

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

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
        print(k)
        if k == 'layers':
            for l in v:
                print('  => %s' % l['name'])

    return model

def build_model(args):
    np.random.seed(args.seed)

    if isinstance(args.n_residual_blocks, int):
        return build_residual_model(args)
    else:
        return build_ordinary_model(args)

