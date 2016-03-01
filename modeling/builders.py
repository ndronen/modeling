import numpy as np

from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (Convolution1D, MaxPooling1D)
from keras.optimizers import (SGD, Adam, Adadelta, Adagrad, RMSprop)
from keras.constraints import maxnorm
from keras.regularizers import l2

from modeling.layers import ImmutableEmbedding, HierarchicalSoftmax

def build_embedding_layer(config, input_width=None):
    try:
        n_embeddings = config.n_vocab
    except AttributeError:
        n_embeddings = config.n_embeddings

    try:
        input_width = config.input_width
    except AttributeError:
        input_width = input_width

    try:
        mask_zero = config.mask_zero
    except AttributeError:
        mask_zero = False

    if hasattr(config, 'embedding_weights') and config.embedding_weights is not None:
        W = np.load(config.embedding_weights)
        if config.train_embeddings is True or config.train_embeddings == 'true':
            return Embedding(n_embeddings, config.n_embed_dims,
                weights=[W], input_length=input_width,
                W_constraint=maxnorm(config.embedding_max_norm),
                mask_zero=mask_zero)
        else:
            return ImmutableEmbedding(n_embeddings, config.n_embed_dims,
                weights=[W], mask_zero=mask_zero,
                input_length=input_width)
    else:
        if config.train_embeddings is True:
            return Embedding(n_embeddings, config.n_embed_dims,
                init=config.embedding_init,
                W_constraint=maxnorm(config.embedding_max_norm),
                mask_zero=mask_zero,
                input_length=input_width)
        else:
            return ImmutableEmbedding(n_embeddings, config.n_embed_dims,
                init=config.embedding_init,
                mask_zero=mask_zero,
                input_length=input_width)

def build_convolutional_layer(config):
    return Convolution1D(config.n_filters, config.filter_width,
        W_constraint=maxnorm(config.filter_max_norm),
        border_mode=config.border_mode,
        W_regularizer=l2(config.l2_penalty))

def build_pooling_layer(config, input_width=None, filter_width=None):
    try:
        input_width = config.input_width
    except AttributeError:
        assert input_width is not None

    try:
        filter_width = config.filter_width
    except AttributeError:
        assert filter_width is not None

    return MaxPooling1D(
        pool_length=input_width - filter_width + 1,
        stride=1)

def build_dense_layer(config, n_hidden=None, activation='linear'):
    if n_hidden is None:
        n_hidden = config.n_hidden
    return Dense(n_hidden,
            W_regularizer=l2(config.l2_penalty),
            W_constraint=maxnorm(config.dense_max_norm),
            activation=activation)

def build_hierarchical_softmax_layer(config):
    # This n_classes is different from the number of unique target values in
    # the training set.  Hierarchical softmax assigns each word to a class
    # and decomposes the softmax into a prediction that's conditioned on
    # class membership.
    return HierarchicalSoftmax(config.n_classes, config.n_hsm_classes,
            batch_size=config.batch_size)

def load_weights(config, model):
    if hasattr(config, 'model_weights') and config.model_weights is not None:
        print('Loading weights from %s' % config.model_weights)
        model.load_weights(config.model_weights)

def build_optimizer(config):
    if config.optimizer == 'SGD':
        optimizer = SGD(lr=config.learning_rate,
            decay=config.decay, momentum=config.momentum,
            clipnorm=config.clipnorm)
    elif config.optimizer == 'Adam':
        optimizer = Adam(clipnorm=config.clipnorm)
    elif config.optimizer == 'RMSprop':
        optimizer = RMSprop(clipnorm=config.clipnorm)
    elif config.optimizer == 'Adadelta':
        optimizer = Adadelta(clipnorm=config.clipnorm)
    elif config.optimizer == 'Adagrad':
        optimizer = Adagrad(clipnorm=config.clipnorm)
    else:
        raise ValueError("don't know how to use optimizer {0}".format(config.optimizer))

    return optimizer
