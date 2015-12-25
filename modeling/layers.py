import logging
import theano.tensor as T
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Layer

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
