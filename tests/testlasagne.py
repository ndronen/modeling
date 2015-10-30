# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import os
import numpy as np

import unittest
import modeling.lasagne_model
import modeling.utils

import theano.tensor as T
import lasagne

# From Lasagne/examples/mnist.py
def load_mnist():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test

class TestModel(modeling.lasagne_model.Classifier):
    def build_input_var(self):
        return T.tensor4('inputs')

    def build_target_var(self):
        return T.ivector('targets')

    def build_updates(self):
        return lasagne.updates.nesterov_momentum(
            self.train_loss, self.params, learning_rate=0.01, momentum=0.9)

    def build_model(self, input_var):
        l_in = lasagne.layers.InputLayer(
                shape=(None, 1, 28, 28), input_var=input_var)

        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

        # Add a fully-connected layer of 800 units, using the linear rectifier, and
        # initializing weights with Glorot's scheme (which is the default anyway).
        l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())

        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

        # Another 800-unit layer.
        l_hid2 = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=800,
                nonlinearity=lasagne.nonlinearities.rectify)

        # 50% dropout again.
        l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        l_out = lasagne.layers.DenseLayer(
                l_hid2_drop, num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)

        # Each layer is linked to its incoming layer(s), so we only need to pass
        # the output layer to give access to a network in Lasagne:
        return l_out
    
class TestLasagneClassifier(unittest.TestCase):
    def test_mnist(self):
        args = {}
        config = modeling.utils.ModelConfig(**args)
        model = TestModel(config)
        X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
        '''
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
        '''
