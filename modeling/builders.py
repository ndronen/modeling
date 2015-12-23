from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import (Convolution1D, MaxPooling1D)
from keras.optimizers import (SGD, Adam, Adadelta, Adagrad, RMSprop)

def build_embedding_layer(args):
    if hasattr(args, 'embedding_weights') and args.embedding_weights is not None:
        W = np.load(args.embedding_weights)
        if args.train_embeddings is True or args.train_embeddings == 'true':
            return Embedding(args.n_embeddings, args.n_embed_dims,
                weights=[W], input_length=args.input_width,
                W_constraint=maxnorm(args.embedding_max_norm))
        else:
            return ImmutableEmbedding(args.n_embeddings, args.n_embed_dims,
                weights=[W], input_length=args.input_width)
    else:
        return Embedding(args.n_embeddings, args.n_embed_dims,
            W_constraint=maxnorm(args.embedding_max_norm),
            input_length=args.input_width)

def build_convolutional_layer(args):
    return Convolution1D(args.n_filters, args.filter_width,
        W_constraint=maxnorm(args.filter_max_norm),
        border_mode=args.border_mode,
        W_regularizer=l2(args.l2_penalty))

def build_pooling_layer(args):
    return MaxPooling1D(
        pool_length=args.input_width - args.filter_width + 1,
        stride=1, ignore_border=False)

def build_dense_layer(args, n_hidden=None, activation='linear'):
    if n_hidden is None:
        n_hidden = args.n_hidden
    return Dense(n_hidden,
            W_regularizer=l2(args.l2_penalty),
            W_constraint=maxnorm(args.dense_max_norm),
            activation=activation)

def load_weights(args, model):
    if hasattr(args, 'model_weights') and args.model_weights is not None:
        print('Loading weights...')
        model.load_weights(args.model_weights)

def build_optimizer(args):
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

    return optimizer
