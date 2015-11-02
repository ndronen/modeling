from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import modeling.lasagne_model
import lasagne

class Model(modeling.lasagne_model.Classifier):
    def build_input_var(self):
        return T.imatrix('inputs')

    def build_target_var(self):
        return T.ivector('targets')

    def build_updates(self):
        return lasagne.updates.nesterov_momentum(
                self.train_loss, self.params,
                learning_rate=0.01, momentum=0.9)

    def build_model(self):
        # Input layer
        input_shape = (self.config.batch_size, self.config.input_width)
        print('input_shape', input_shape)
        model = lasagne.layers.InputLayer(shape=input_shape,
                input_var=self.input_var)
    
        # Embedding layer
        model = lasagne.layers.EmbeddingLayer(model,
                self.config.n_vocab, self.config.n_word_dims)
    
        # Convolutional layer
        model = lasagne.layers.Conv1DLayer(model,
                num_filters=self.config.n_filters,
                filter_size=self.config.filter_width,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())

        print('pool_size', self.config.input_width-self.config.filter_width-1)
    
        # Max-pooling layer 
        model = lasagne.layers.MaxPool1DLayer(model,
                pool_size=self.config.input_width-self.config.filter_width-1)

        # Flatten layer
        #model = lasagne.layers.FlattenLayer(model)
    
        # Fully-connected layer
        model = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(model, p=.0),
                num_units=self.config.n_filters*2,
                nonlinearity=lasagne.nonlinearities.rectify)
    
        # Output layer 
        model = lasagne.layers.DenseLayer(
                lasagne.layers.dropout(model, p=.5),
                num_units=self.config.n_classes,
                nonlinearity=lasagne.nonlinearities.softmax)
    
        return model
