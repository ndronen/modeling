import unittest
import numpy as np
import theano
import theano.tensor as T
#import theano.tensor.nnet 

import keras.models
import keras.layers.core

from modeling.layers import HierarchicalSoftmax
import modeling.utils
import modeling.builders

class TestHierarchicalSoftmax(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1
        self.input_dim = 4
        self.n_hsm_classes = 5
        self.n_outputs_per_class = 3
        self.output_size = self.n_hsm_classes * self.n_outputs_per_class

    def test_hierarchical_softmax_integrated(self):
        net = keras.models.Sequential()
        net.add(keras.layers.core.Dense(100, input_dim=self.input_dim, activation='relu'))
        net.add(HierarchicalSoftmax(
            self.n_hsm_classes, self.n_outputs_per_class,
            batch_size=self.batch_size))
        net.compile(loss='categorical_crossentropy', optimizer='Adam')
        x = np.random.normal(size=(self.batch_size, self.input_dim))
        target = net.predict_proba(x, verbose=0)
        n_classes = self.n_hsm_classes * self.n_outputs_per_class
        self.assertEqual((self.batch_size, n_classes), target.shape)

    def test_hierarchical_softmax_isolated(self):
        layer = HierarchicalSoftmax(self.n_hsm_classes, self.n_outputs_per_class,
                batch_size=self.batch_size,
                input_dim=self.input_dim)
        layer.build()

        xt = T.matrix('x')
        f = theano.function([xt], layer._get_output(xt))
        x = np.random.normal(size=(self.batch_size, self.input_dim)).astype(np.float32)

        output = f(x)
        self.assertTrue(output.shape == (self.batch_size, self.output_size))
        self.assertTrue(np.allclose(1.0, output.sum()))

    #@unittest.skip('')
    def test_theano_h_softmax(self):
        """
        Tests the output dimensions of the h_softmax when a target is provided or
        not.

        This test came from 
        """

        #############
        # Initialize shared variables
        #############
    
        floatX = theano.config.floatX
        shared = theano.shared
    
        # Class softmax.
        W1 = np.asarray(np.random.normal(
            size=(self.input_dim, self.n_hsm_classes)), dtype=floatX)
        W1 = shared(W1)
        b1 = np.asarray(np.zeros((self.n_hsm_classes,)), dtype=floatX)
        b1 = shared(b1)
    
        # Class member softmax.
        W2 = np.asarray(np.random.normal(
            size=(self.n_hsm_classes, self.input_dim, self.n_outputs_per_class)),
            dtype=floatX)
        W2 = shared(W2)
        b2 = np.asarray(
            np.zeros((self.n_hsm_classes, self.n_outputs_per_class)), dtype=floatX)
        b2 = shared(b2)
    
        #############
        # Build graph
        #############
        x = T.matrix('x')
        y = T.ivector('y')
    
        # This only computes the output corresponding to the target
        y_hat_tg = theano.tensor.nnet.h_softmax(x,
                self.batch_size, self.output_size, self.n_hsm_classes, self.n_outputs_per_class,
                W1, b1, W2, b2, y)
    
        # This computes all the outputs
        y_hat_all = theano.tensor.nnet.h_softmax(x,
                self.batch_size, self.output_size, self.n_hsm_classes, self.n_outputs_per_class,
                W1, b1, W2, b2)
    
        #############
        # Compile functions
        #############
        fun_output_tg = theano.function([x, y], y_hat_tg)
        fun_output = theano.function([x], y_hat_all)
    
        #############
        # Test
        #############
        x_mat = np.random.normal(size=(self.batch_size, self.input_dim)).astype(floatX)
        y_mat = np.random.randint(0, self.output_size, self.batch_size).astype('int32')
        
        self.assertTrue(fun_output_tg(x_mat, y_mat).shape == (self.batch_size,))
        self.assertTrue(fun_output(x_mat).shape == (self.batch_size, self.output_size))
    
if __name__ == '__main__':
    unittest.main()
