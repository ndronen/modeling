import math
import numpy as np
import theano
import theano.tensor as T
import unittest
import logging

logger = logging.getLogger()

from keras.layers.core import Layer
from keras.utils.theano_utils import sharedX

class SplitOutputByFilter(Layer):
    """
    input: (batch_size, max_seq_len, n_filters * filter_width)
    output: (batch_size, n_filters, max_seq_len, filter_width)
    """
    def __init__(self, n_filters, filter_width):
        super(SplitOutputByFilter, self).__init__()
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.input = T.tensor3()

    def slice(self, i, X):
        start = i * self.filter_width
        end = (i+1) * self.filter_width
        return X[:, :, start:end]

    def _get_output(self, X):
        outputs, updates = theano.scan(
                fn=self.slice,
                outputs_info=None,
                sequences=[T.arange(self.n_filters)],
                non_sequences=X)
        return outputs.dimshuffle(1, 0, 2, 3)

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def get_config(self):
        return {"name": self.__class__.__name__}

class SlidingWindowL2MaxPooling(Layer):
    '''
    input: (batch_size, n_filters, max_seq_len, filter_width)
    output: (batch_size, n_filters, filter_width, filter_width)
    '''
    def __init__(self, batch_size, n_filters, filter_width, max_seq_len):
        super(SlidingWindowL2MaxPooling, self).__init__()
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.max_seq_len = max_seq_len

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def _get_output(self, X):
        outputs, updates = theano.scan(
                fn=self.sample_dimension,
                sequences=[T.arange(self.batch_size)],
                non_sequences=X)
        return outputs

    def sample_dimension(self, i, X):
        '''
        Takes a 4-tensor of shape `(batch_size, n_filters, max_seq_len,
        filter_width)` and an index into its first dimension.  Returns the
        `(batch_size, n_filters, filter_width, filter_width)` subtensor
        with the greatest L2 norm along the third dimension.

        Parameters
        ----------
        X : a 4-tensor
            An `(batch_size, n_filters, max_seq_len, filter_width)` tensor.
        i : int
            An index into the first dimension of `X`.

        Returns
        ----------
        A 3-tensor of shape `(n_filters, filter_width, filter_width)`
        consisting of the subtensor of `X` with the greatest L2 norm along
        `X`'s third dimension (where `max_seq_len` lies).
        '''
        outputs, updates = theano.scan(
                fn=self.filter_dimension,
                sequences=[T.arange(self.n_filters)],
                non_sequences=X[i, :, :, :])

        return outputs

    def filter_dimension(self, i, X):
        '''
        Takes a 3-tensor of shape `(n_filters, max_seq_len, filter_width)`
        and an index into its first dimension.  Returns the
        `(filter_width, filter_width)` subtensor of `X` with the greatest
        L2 norm along the second dimension.

        Parameters
        ----------
        X : a 3-tensor
            An `(batch_size, n_filters, max_seq_len, filter_width)` tensor.
        i : int
            An index into the first dimension of `X`.

        Returns
        ----------
        A 2-tensor of shape `(filter_width, filter_width)` consisting
        of the subtensor of the i-th element along the first dimension
        of `X` with the greatest L2 norm along `X`'s second dimension
        (where `max_seq_len` lies).
        '''
        norms, updates = theano.scan(
                fn=self.norm,
                sequences=[T.arange(self.max_seq_len)],
                non_sequences=X[i, :, :])
        start_window = T.argmax(norms)
        end_window = start_window + self.filter_width
        return X[i, start_window:end_window, :]

    def norm(self, i, X):
        return (X[i:i+self.filter_width, :] ** 2).sum()

class ZeroFillDiagonals(Layer):
    '''
    input: (batch_size, n_filters, filter_width, filter_width)
    output: (batch_size, n_filters, filter_width, filter_width) with the
    diagonal of the last two `(filter_width, filter_width)` dimensions 
    zeroed out.
    '''
    def __init__(self, batch_size, n_filters, filter_width):
        super(ZeroFillDiagonals, self).__init__()
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.filter_width = filter_width

        # Construct a shared boolean matrix by which to multiply the input
        # element-wise.  It should be 0 everywhere except on the diagonals
        # of the last two dimensions.
        input_shape = (batch_size, n_filters, filter_width, filter_width)
        mask = np.ones(input_shape)
        diag_indices = np.arange(filter_width)
        for i in np.arange(batch_size):
            for j in np.arange(n_filters):
                mask[i, j, diag_indices, diag_indices] = 0
        self.mask = sharedX(mask, dtype='int32')

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def _get_output(self, X):
        return X * self.mask
