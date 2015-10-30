# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
import numpy as np
from theano import function
import theano.tensor as T

from keras.layers.core import Layer

class TemporalDifference(Layer):
    """
    Given a 3-tensor with shape (nb_samples, maxlen, output_dim), outputs
    the difference X[
    """
    def _get_output(self, X):
        return X[:, 1:, :] - X[:, 0:X.shape[1]-1, :]

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def get_config(self):
        return {"name": self.__class__.__name__}

class TestTemporalDifference(unittest.TestCase):
    def testForward(self):
        nb_examples = 2
        maxlen = 7
        output_dim = nb_word_dim = 5
        x = np.random.normal(size=(nb_examples, maxlen, output_dim)).astype(np.float32)
        expected = x[:, 1:, :] - x[:, 0:x.shape[1]-1, :]
        X = T.tensor3('X')
        retval = TemporalDifference()._get_output(X)
        f = function([X], retval)
        actual = f(x)
        self.assertTrue(np.allclose(actual, expected))
