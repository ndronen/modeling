import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop

from modeling.layers import ImmutableEmbedding
from modeling.difference import TemporalDifference

def build_model(args):
    np.random.seed(args.seed)

    model = Sequential()

    if hasattr(args, 'embedding_weights') and args.embedding_weights is not None:
        W = np.load(args.embedding_weights)
        if args.train_embeddings:
            model.add(Embedding(args.n_vocab, args.n_word_dims,
                weights=[W],
                W_constraint=maxnorm(args.embedding_max_norm)))
        else:
            model.add(ImmutableEmbedding(args.n_vocab, args.n_word_dims,
                weights=[W]))
    else:
        model.add(Embedding(args.n_vocab, args.n_word_dims,
            W_constraint=maxnorm(args.embedding_max_norm)))

    if args.use_difference:
        model.add(TemporalDifference())

    model.add(Convolution1D(args.n_word_dims, args.n_filters, args.filter_width,
        W_constraint=maxnorm(args.filter_max_norm),
        border_mode=args.border_mode,
        W_regularizer=l2(args.l2_penalty)))
    #if 'normalization' in args.regularization_layer:
    #    model.add(BatchNormalization(
    #        (args.input_width-args.filter_width+1, args.n_filters)))
    model.add(Activation('relu'))

    model.add(MaxPooling1D(
        pool_length=args.input_width - args.filter_width + 1,
        stride=None, ignore_border=False))
    model.add(Flatten())
    if 'dropout' in args.regularization_layer:
        model.add(Dropout(args.dropout_p_conv))
    if 'normalization' in args.regularization_layer:
        model.add(BatchNormalization((args.n_filters,)))

    model.add(Dense(args.n_filters, 2*args.n_filters,
        W_regularizer=l2(args.l2_penalty)))
    model.add(Activation('relu'))
    if 'dropout' in args.regularization_layer:
        model.add(Dropout(args.dropout_p))
    if 'normalization' in args.regularization_layer:
        model.add(BatchNormalization((2*args.n_filters,)))

    model.add(Dense(2*args.n_filters, 2*args.n_filters))
    model.add(Activation('relu'))
    if 'dropout' in args.regularization_layer:
        model.add(Dropout(args.dropout_p))
    if 'normalization' in args.regularization_layer:
        model.add(BatchNormalization((2*args.n_filters,)))

    model.add(Dense(2*args.n_filters, 2*args.n_filters,
        W_regularizer=l2(args.l2_penalty)))
    model.add(Activation('relu'))
    if 'dropout' in args.regularization_layer:
        model.add(Dropout(args.dropout_p))
    if 'normalization' in args.regularization_layer:
        model.add(BatchNormalization((2*args.n_filters,)))

    model.add(Dense(2*args.n_filters, args.n_classes,
        W_regularizer=l2(args.l2_penalty)))
    #if 'normalization' in args.regularization_layer:
    #    model.add(BatchNormalization((args.n_classes,)))
    model.add(Activation('softmax'))

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.learning_rate,
            decay=args.decay, momentum=args.momentum,
            clipnorm=args.clipnorm)
    elif args.optimizer == 'Adam':
        optimizer = Adam(clipnorm=args.clipnorm)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(clipnorm=args.clipnorm)
    elif args.optimizer == 'Adadelta':
        optimizer = Adadelta(clipnorm=args.clipnorm)
    elif args.optimizer == 'Adagrad':
        optimizer = Adagrad(clipnorm=args.clipnorm)
    else:
        raise ValueError("don't know how to use optimizer {0}".format(args.optimizer))

    model.compile(loss=args.loss, optimizer=optimizer)

    return model
