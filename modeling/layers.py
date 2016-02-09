import unittest
import logging
import numpy as np
import theano.tensor as T
import theano.tensor.nnet 

from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Layer
from keras import activations, initializations, regularizers, constraints

from keras import backend as K

logger = logging.getLogger()

class ImmutableEmbedding(Embedding):
    '''
    Same as Embedding except the weights are not parameters of the
    network.  This can be useful when the layer is initialized with
    pre-trained embeddings, such as Word2Vec.

    @input_dim: size of vocabulary (highest input integer + 1)
    @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, **kwargs):
        super(ImmutableEmbedding, self).__init__(
                input_dim, output_dim, **kwargs)
        self.params = []

    def build(self):
        super(ImmutableEmbedding, self).build()
        self.params = []

class ImmutableConvolution1D(Convolution1D):
    '''
    Same as Convolution1D except the convolutional filters are not 
    parameters of the network.  This can be useful when the layer 
    is initialized with pre-trained convolutional filters.

    @nb_filters: the number of convolutional filters
    @filter_width: the width of each filter
    '''
    def __init__(self, nb_filters, filter_width, **kwargs):
        super(ImmutableConvolution1D, self).__init__(
                nb_filters, filter_width, **kwargs)
        self.params = []

    def build(self):
        super(ImmutableConvolution1D, self).build()
        self.params = []

class Transpose(Layer):
    def __init__(self):
        super(Transpose, self).__init__()
        self.input = T.matrix()

    def _get_output(self, X):
        return X.T

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def get_config(self):
        return {"name": self.__class__.__name__}

class HierarchicalSoftmax(Layer):
    def __init__(self, output_dim, nb_hsm_classes, batch_size,
            init='glorot_uniform',
            W1_weights=None, W1_regularizer=None, W1_constraint=None,
            W2_weights=None, W2_regularizer=None, W2_constraint=None,
            b1_regularizer=None, b1_constraint=None,
            b2_regularizer=None, b2_constraint=None,
            input_dim=None, **kwargs):

        self.__dict__.update(locals())
        del self.self

        self.init = initializations.get(init)
        #self.output_dim = nb_classes * nb_outputs_per_class
        self.nb_outputs_per_class = int(np.ceil(output_dim / float(nb_hsm_classes)))

        self.W1_regularizer = regularizers.get(W1_regularizer)
        self.b1_regularizer = regularizers.get(b1_regularizer)
        self.W2_regularizer = regularizers.get(W2_regularizer)
        self.b2_regularizer = regularizers.get(b2_regularizer)

        self.W1_constraint = constraints.get(W1_constraint)
        self.b1_constraint = constraints.get(b1_constraint)
        self.W2_constraint = constraints.get(W2_constraint)
        self.b2_constraint = constraints.get(b2_constraint)

        self.constraints = [self.W1_constraint, self.b1_constraint,
                self.W2_constraint, self.b2_constraint]

        #self.initial_weights = weights
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        self.input = T.matrix()
        super(HierarchicalSoftmax, self).__init__(**kwargs)

    def build(self):
        #print('self.input_shape', self.input_shape)
        n_features = self.input_shape[1]

        self.W1 = self.init((n_features, self.nb_hsm_classes))
        self.b1 = K.zeros((self.nb_hsm_classes,))

        self.W2 = self.init((self.nb_hsm_classes, n_features, self.nb_outputs_per_class))
        self.b2 = K.zeros((self.nb_hsm_classes, self.nb_outputs_per_class))

        self.trainable_weights = [self.W1, self.b1,
                self.W2, self.b2]
        
        self.regularizers = []
        if self.W1_regularizer:
            self.W1_regularizer.set_param(self.W1)
            self.regularizers.append(self.W1_regularizer)
        
        if self.b1_regularizer:
            self.b1_regularizer.set_param(self.b1)
            self.regularizers.append(self.b1_regularizer)

        if self.W2_regularizer:
            self.W2_regularizer.set_param(self.W2)
            self.regularizers.append(self.W2_regularizer)
        
        if self.b2_regularizer:
            self.b2_regularizer.set_param(self.b2)
            self.regularizers.append(self.b2_regularizer)

    @property
    def output_shape(self):
        print('HierarchicalSoftmax.output_shape', self.input_shape[0], self.output_dim)
        return (self.input_shape[0], self.output_dim)

    def _get_output(self, X):
        output = theano.tensor.nnet.h_softmax(X,
                #self.input_shape[1], self.output_dim,
                self.batch_size, self.output_dim,
                self.nb_hsm_classes, self.nb_outputs_per_class,
                self.W1, self.b1,
                self.W2, self.b2)
        return output

    def get_output(self, train=False):
        return self._get_output(self.get_input(train))

