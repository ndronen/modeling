import sys
import h5py
sys.setrecursionlimit(5000)
import json
import h5py

from sklearn.utils import check_random_state

import numpy as np

from keras.models import Sequential, Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten, Layer
from keras.layers.normalization import BatchNormalization

import modeling.data
from modeling.builders import (build_embedding_layer,
    build_convolutional_layer, build_pooling_layer,
    build_dense_layer, build_optimizer, load_weights,
    build_hierarchical_softmax_layer)
from modeling.utils import balanced_class_weights

class HDF5FileDataset(object):
    def __init__(self, file_path, data_name, target_name, batch_size, one_hot=True, random_state=17):
        assert isinstance(data_name, (list,tuple))
        assert isinstance(target_name, (list,tuple))

        random_state = check_random_state(random_state)

        self.__dict__.update(locals())
        del self.self

        self._load_data()
        self._check_data()

    def _load_data(self):
        self.hdf5_file = h5py.File(self.file_path)
        self.n_classes = {}
        for target_name in self.target_name:
            self.n_classes[target_name] = np.max(self.hdf5_file[target_name])+1

    def _check_data(self):
        self.n = None
        for data_name in self.data_name:
            if self.n is None:
                self.n = len(self.hdf5_file[data_name])
            else:
                assert len(self.hdf5_file[data_name]) == self.n
        for target_name in self.target_name:
            assert len(self.hdf5_file[target_name]) == self.n

    def __getitem__(self, name):
        return self.hdf5_file[name].value

    def class_weights(self, class_weight_exponent, target='multiclass_correction_target'):
        return balanced_class_weights(
                self.hdf5_file[target],
                2,
                class_weight_exponent)

    def generator(self, one_hot=None, batch_size=None):
        if one_hot is None: one_hot = self.one_hot
        if batch_size is None: batch_size = self.batch_size

        while 1:
            idx = self.random_state.choice(self.n, size=batch_size, replace=False)
            batch = {}
            for data_name in self.data_name:
                batch[data_name] = self.hdf5_file[data_name].value[idx]
            for target_name in self.target_name:
                target = self.hdf5_file[target_name].value[idx]
                if one_hot:
                    batch[target_name] = np_utils.to_categorical(target,
                            self.n_classes[target_name])
                else:
                    batch[target_name] = target

            yield batch

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def add_bn_relu(graph, args, prev_layer):
    bn_name = prev_layer + '_bn'
    relu_name = prev_layer + '_relu'
    if args.batch_normalization:
        graph.add_node(BatchNormalization(), name=bn_name, input=prev_layer)
        prev_layer = bn_name
    graph.add_node(Activation('relu'), name=relu_name, input=prev_layer)
    return relu_name

def build_model(args, train_data):
    np.random.seed(args.seed)

    graph = Graph()

    non_word_input = 'non_word_marked_chars'
    non_word_input_width = train_data[non_word_input].shape[1]

    graph.add_input(non_word_input, input_shape=(non_word_input_width,), dtype='int')
    graph.add_node(build_embedding_layer(args, input_width=non_word_input_width),
            name='non_word_embedding', input=non_word_input)
    graph.add_node(build_convolutional_layer(args), name='non_word_conv', input='non_word_embedding')
    non_word_prev_layer = add_bn_relu(graph, args, 'non_word_conv')
    graph.add_node(build_pooling_layer(args, input_width=non_word_input_width),
            name='non_word_pool', input=non_word_prev_layer)
    graph.add_node(Flatten(), name='non_word_flatten', input='non_word_pool')

    # Add some number of fully-connected layers without skip connections.
    prev_layer = 'non_word_flatten'
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

    n_classes = np.max(train_data['multiclass_correction_target']) + 1
    if hasattr(args, 'n_hsm_classes'):
        graph.add_node(build_hierarchical_softmax_layer(args),
            name='softmax', input=prev_layer)
    else:
        graph.add_node(build_dense_layer(args, n_classes,
            activation='softmax'), name='softmax', input=prev_layer)

    graph.add_output(name='multiclass_correction_target', input='softmax')

    load_weights(args, graph)

    optimizer = build_optimizer(args)

    graph.compile(loss={'multiclass_correction_target': args.loss}, optimizer=optimizer)

    return graph

def fit(config, callbacks=[]):
    train_data = HDF5FileDataset(
            config.train_path,
            config.data_name,
            [config.target_name],
            config.batch_size,
            config.seed)

    validation_data = HDF5FileDataset(
            config.validation_path,
            config.data_name,
            [config.target_name],
            config.batch_size,
            config.seed)

    graph = build_model(config, train_data)

    graph.fit_generator(train_data.generator(),
            samples_per_epoch=int(train_data.n/100),
            nb_epoch=config.n_epochs,
            validation_data=validation_data.generator(),
            nb_val_samples=10000,
            callbacks=callbacks,
            class_weight=train_data.class_weights(config.class_weight_exponent))
