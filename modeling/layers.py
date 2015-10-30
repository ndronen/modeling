import logging
import theano.tensor as T
from keras.layers.embeddings import Embedding
from keras.layers.core import Layer

logger = logging.getLogger()

class ImmutableEmbedding(Embedding):
    '''
        Same as keras.layers.Embedding except the weights are parameters
        of the network.  This can be useful when the layer is initialized
        with pre-trained embeddings, such as Word2Vec.

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, **kwargs):

        super(ImmutableEmbedding, self).__init__(
                input_dim, output_dim, **kwargs)

        print("W", self.get_weights())
        print("params", self.params)
        self.params = []
        print("params", self.params)


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
