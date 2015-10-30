#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
import argparse
import h5py

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils

from outliers import PMeansMultivariateNormal

def create_dataset(n, train_size, valid_size):
    means = np.arange(100)
    cov = [range(1, 101)] * 100
    mvn = PMeansMultivariateNormal(means, cov, (n,))
    X = mvn.generate()

    assert n % 2 == 0
    assert n > train_size + valid_size

    # Make the data different along one dimension.
    even = np.arange(0, n, step=2)
    X[even, 0] = np.random.uniform(-.25, 1.75, size=n/2)
    # Make each odd-numbered row the inverse of its previous row.
    X[even+1, 0] = np.random.uniform(-1.75, .25, size=n/2)

    X += np.random.uniform(0.01, size=X.shape)
    X = X.astype(np.float32)

    y = np.array([[0,1] * (n/2)]).reshape((n,1))
    y = y.astype(np.int32)

    X_train = X[0:train_size, :]
    X_valid = X[train_size:train_size+valid_size, :]
    X_test = X[train_size+valid_size:, :]

    y_train = y[0:train_size]
    y_valid = y[train_size:train_size+valid_size]
    y_test = y[train_size+valid_size:]

    return X_train, X_valid, X_test, \
            y_train, y_valid, y_test


def build_model(n_inputs, n_hidden, n_classes):
    model = Sequential()
    model.add(Dense(n_inputs, n_hidden))
    model.add(BatchNormalization((n_hidden,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, n_hidden))
    model.add(BatchNormalization((n_hidden,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, n_hidden))
    model.add(BatchNormalization((n_hidden,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, n_hidden))
    model.add(BatchNormalization((n_hidden,)))
    model.add(Activation('relu'))
    model.add(Dense(n_hidden, n_classes))
    model.add(Activation('softmax'))
    
    optimizer = Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model

def get_parser():
    parser = argparse.ArgumentParser(
            description='train a model to demonstrate contrasting cases')
    parser.add_argument(
            '--shuffle', action='store_true',
            help='shuffle the training examples after each epoch (i.e. do not use contrasting cases)')
    parser.add_argument(
            '--n', type=int, default=10000,
            help='the size of the data set to create')
    parser.add_argument(
            '--train-size', type=int, default=7000,
            help='the number of examples from the data set to allocate to training')
    parser.add_argument(
            '--valid-size', type=int, default=1500,
            help='the number of examples from the data set to allocate to validation')
    parser.add_argument(
            '--batch-size', type=int, default=10,
            help='mini-batch size')
    parser.add_argument(
            '--n-epochs', type=int, default=20,
            help='number of epochs to train')
    parser.add_argument(
            '--verbose', action='store_true',
            help='print progress')
    
    return parser.parse_args()

def main(args):
    x_train, x_valid, x_test, \
            y_train, y_valid, y_test = create_dataset(
                    args.n, args.train_size, args.valid_size)

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_valid = y_valid.reshape((y_valid.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    n_classes = len(np.unique(y_train))

    '''
    print('y_train', y_train.shape)
    print('y_valid', y_valid.shape)
    print('y_test', y_test.shape)
    print('n_classes', n_classes, np.unique(y_train))
    '''
    
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(
            y_train, n_classes).astype(np.int32)
    y_valid = np_utils.to_categorical(
            y_valid, n_classes).astype(np.int32)
    y_test = np_utils.to_categorical(
            y_test, n_classes).astype(np.int32)
    
    if args.shuffle:
        print('Training (shuffled)')
        # Leave odd-numbered rows where they are; shuffle only
        # even-numbered ones.  This ensures that each minibatch has one
        # example from each class.
        perm = np.arange(x_train.shape[0])
        evens = np.arange(0, x_train.shape[0], 2)
        perm[evens] = np.random.permutation(evens)
    else:
        print('Training (contrasting cases)')
    
    model = build_model(100, 20, n_classes)

    print('x_train', x_train.dtype)
    print('y_train', y_train.dtype)
    
    model.fit(x_train, y_train,
            batch_size=args.batch_size,
            shuffle=False,
            nb_epoch=args.n_epochs,
            show_accuracy=True,
            verbose=2 if args.verbose else 0,
            validation_data=(x_valid, y_valid))
    
    score = model.evaluate(x_test, y_test,
            show_accuracy=True,
            verbose=1 if args.verbose else 0)
    
    if args.shuffle:
        print('Test accuracy (shuffled)', score[1])
    else:
        print('Test accuracy (contrasting cases)', score[1])
    
if __name__ == '__main__':
    sys.exit(main(get_parser()))
