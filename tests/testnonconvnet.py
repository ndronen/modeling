import unittest
import random
import numpy as np
import theano
import theano.tensor as T

from keras import models
from keras.layers import embeddings
from keras.layers import core

from modeling.nonconvnet import ZeroFillDiagonals, \
        SplitOutputByFilter, \
        SlidingWindowL2MaxPooling

class TestNonConvNet(unittest.TestCase):
    def setUp(self):
        self.n_vocab = 100
        self.n_word_dims = 5
        self.filter_width = 4
        self.n_filters = 3
        self.max_seq_len = 9
        self.batch_size = 3

    def setSeeds(self):
        np.random.seed(1)

    def testNonConvNet(self):
        self.setSeeds()

        x = np.random.randint(self.n_vocab, size=(self.batch_size,
                self.max_seq_len))

        model = models.Sequential()

        # input: (batch_size, max_seq_len)
        # output: (batch_size, max_seq_len, n_word_dims)
        model.add(embeddings.Embedding(self.n_vocab, self.n_word_dims))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l1 = (self.batch_size, self.max_seq_len,
                self.n_word_dims)
        output_l1 = model.predict(x)
        self.assertEqual(expected_shape_l1, output_l1.shape)

        # input: (batch_size, max_seq_len, n_word_dims)
        # output: (batch_size, max_seq_len, n_filters * filter_width)
        model.add(core.TimeDistributedDense(
            self.n_word_dims, self.n_filters * self.filter_width))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l2 = (self.batch_size, self.max_seq_len,
                self.n_filters * self.filter_width)
        output_l2 = model.predict(x)
        self.assertEqual(expected_shape_l2, output_l2.shape)

        # input: (batch_size, max_seq_len, n_filters * filter_width)
        # output: (batch_size, n_filters, max_seq_len, filter_width)
        model.add(SplitOutputByFilter(self.n_filters, self.filter_width))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l3 = (self.batch_size, self.n_filters,
                self.max_seq_len, self.filter_width)
        output_l3 = model.predict(x)
        self.assertEqual(expected_shape_l3, output_l3.shape)

        # input: (batch_size, n_filters, max_seq_len, filter_width)
        # output: (batch_size, n_filters, filter_width, filter_width)
        model.add(SlidingWindowL2MaxPooling(
                self.batch_size, self.n_filters,
                self.filter_width, self.max_seq_len))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l4 = (self.batch_size, self.n_filters,
                self.filter_width, self.filter_width)
        output_l4 = model.predict(x)
        self.assertEqual(expected_shape_l4, output_l4.shape)

        # input: (batch_size, n_filters, filter_width, filter_width)
        # output: (batch_size, n_filters, filter_width, filter_width)
        model.add(ZeroFillDiagonals(
                self.batch_size, self.n_filters, self.filter_width))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l5 = (self.batch_size, self.n_filters,
                self.filter_width, self.filter_width)
        output_l5 = model.predict(x)
        self.assertEqual(expected_shape_l5, output_l5.shape)

    def testSplitOutputByFilter(self):
        self.setSeeds()

        input_shape = (self.batch_size, self.max_seq_len,
                self.n_filters * self.filter_width)
        output_shape = (self.batch_size, self.n_filters,
                self.max_seq_len, self.filter_width)

        x = np.arange(np.prod(input_shape))
        x = x.reshape(input_shape).astype(np.int32)
        y = np.zeros_like(x)
        y = np.reshape(y, output_shape)

        for i in range(self.n_filters):
            s = x[:, :, i*self.filter_width:(i+1)*self.filter_width]
            y[:, i, :, :] = s

        xt = T.itensor3('xt')
        layer = SplitOutputByFilter(self.n_filters, self.filter_width)
        yt = layer._get_output(xt)

        f = theano.function(inputs=[xt], outputs=yt)
        y_theano = f(x)

        self.assertEquals(y.shape, y_theano.shape)
        self.assertTrue(np.all(y == y_theano))

    def testSlidingWindowL2MaxPooling(self):
        self.assertTrue(
                self.max_seq_len - self.filter_width > self.n_filters)

        self.setSeeds()

        input_shape = (self.batch_size, self.n_filters,
                self.max_seq_len, self.filter_width)
        output_shape = (self.batch_size, self.n_filters,
                self.filter_width, self.filter_width)

        x = np.zeros(shape=input_shape)
        expected = np.zeros(shape=output_shape)

        max_input_shape = (self.batch_size, self.filter_width, self.filter_width)

        # For the i-th filter, make i the offset at which the maximum
        # L2 norm occurs.
        for i in np.arange(self.n_filters):
            start = i
            end = i+self.filter_width
            values = i + np.arange(np.prod(max_input_shape))
            values = values.reshape(max_input_shape)
            x[:, i, start:end, :] = values
            expected[:, i, :, :] = values

        it = T.iscalar()
        x3d = T.dtensor3('x3d')
        x4d = T.dtensor4('x4d')

        layer = SlidingWindowL2MaxPooling(
                self.batch_size, self.n_filters, self.filter_width,
                self.max_seq_len)

        '''
        Use the first sample and first filter to test `filter_dimension`.
        '''
        yt_filter_dim = layer.filter_dimension(it, x3d)
        f_filter_dim = theano.function(inputs=[it, x3d], outputs=yt_filter_dim)
        y_filter_dim_out = f_filter_dim(0, x[0])
        self.assertEquals((self.filter_width, self.filter_width),
                y_filter_dim_out.shape)
        self.assertTrue(np.all(expected[0, 0, :, :] == y_filter_dim_out))

        '''
        Use the first sample to test `filter_dimension`.
        '''
        yt_sample_dim = layer.sample_dimension(it, x4d)
        f_sample_dim = theano.function(inputs=[it, x4d], outputs=yt_sample_dim)
        y_sample_dim_out = f_sample_dim(0, x)
        self.assertEquals((self.n_filters, self.filter_width, self.filter_width),
                y_sample_dim_out.shape)
        self.assertTrue(np.all(expected[0, :, :, :] == y_sample_dim_out))

        '''
        Use all of `x` to test `_get_output`.
        '''
        yt_output = layer._get_output(x4d)
        f_output = theano.function(inputs=[x4d], outputs=yt_output)
        yt_out = f_output(x)
        self.assertEquals(
                (self.batch_size, self.n_filters, self.filter_width,
                self.filter_width), yt_out.shape)
        self.assertTrue(np.all(expected == yt_out))

    def testZeroFillDiagonals(self):
        input_shape = (self.batch_size, self.n_filters,
                self.filter_width, self.filter_width)
        mask = np.ones(input_shape)
        diag_indices = np.arange(self.filter_width)
        for i in np.arange(self.batch_size):
            for j in np.arange(self.n_filters):
                mask[i, j, diag_indices, diag_indices] = 0

        x = np.arange(np.prod(input_shape)).reshape(input_shape)
        expected = x * mask

        x4d = T.dtensor4('x4d')
        layer = ZeroFillDiagonals(
                self.batch_size, self.n_filters, self.filter_width)
        yt_output = layer._get_output(x4d)
        f_output = theano.function(inputs=[x4d], outputs=yt_output)

        yt_out = f_output(x)
        self.assertEquals(expected.shape, yt_out.shape)
        self.assertTrue(np.all(expected == yt_out))
