import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad

from modeling.layers import ImmutableEmbedding

def build_model(args):
    print("args", vars(args))

    model = Sequential()

    np.random.seed(args.seed)

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
            mask_zero=args.mask_zero,
            W_constraint=maxnorm(args.embedding_max_norm)))

    model.add(LSTM(args.n_word_dims, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True))
    if args.regularization_layer == 'dropout':
        model.add(Dropout(0.2))
    #elif args.regularization_layer == 'normalization':
    #    model.add(BatchNormalization((args.n_filters,)))

    model.add(LSTM(args.n_units, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True))
    if args.regularization_layer == 'dropout':
        model.add(Dropout(0.2))
    #elif args.regularization_layer == 'normalization':
    #    model.add(BatchNormalization((args.n_filters,)))

    '''
    model.add(LSTM(args.n_units, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True))
    if args.regularization_layer == 'dropout':
        model.add(Dropout(0.2))
    #elif args.regularization_layer == 'normalization':
    #    model.add(BatchNormalization((args.n_filters,)))
    '''

    model.add(LSTM(args.n_units, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=False))
    if args.regularization_layer == 'dropout':
        model.add(Dropout(0.2))
    #elif args.regularization_layer == 'normalization':
    #    model.add(BatchNormalization((args.n_filters,)))

    model.add(Dense(args.n_units, args.n_classes,
        W_regularizer=l2(args.l2_penalty)))
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
