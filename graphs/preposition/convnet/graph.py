import numpy as np

from keras.models import Sequential, Graph
from keras.layers.core import Dense, TimeDistributedMerge, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop

from modeling.layers import ImmutableEmbedding
from modeling.difference import TemporalDifference

def build_graph(args):
    print("args", vars(args))

    np.random.seed(args.seed)

    graph = Graph()

    graph.add_input(name=args.variable_window_name, ndim=2, dtype=int)
    #graph.add_input(name=args.fixed_window_name, ndim=2, dtype=int)

    #######################################################################
    # The variable-width window (i.e. entire sentence) follows this path
    # through the graph.
    #######################################################################

    if hasattr(args, 'embedding_weights') and args.embedding_weights is not None:
        W = np.load(args.embedding_weights)
        if args.train_embeddings:
            embedding = Embedding(args.n_vocab, args.n_word_dims,
                    weights=[W],
                    W_constraint=maxnorm(args.embedding_max_norm))
        else:
            embedding = ImmutableEmbedding(args.n_vocab, args.n_word_dims,
                weights=[W])
    else:
        embedding = Embedding(args.n_vocab, args.n_word_dims,
            W_constraint=maxnorm(args.embedding_max_norm))

    graph.add_node(embedding,
            name='embedding', input=args.variable_window_name)

    conv = Convolution1D(args.n_word_dims, args.n_filters, args.filter_width,
            activation='relu',
            W_constraint=maxnorm(args.filter_max_norm),
            border_mode=args.border_mode,
            W_regularizer=l2(args.l2_penalty))
    graph.add_node(conv, name='conv', input='embedding')

    pool = MaxPooling1D(
            pool_length=args.variable_input_width - args.filter_width + 1,
            stride=None, ignore_border=False)
    graph.add_node(pool, name='pool', input='conv')

    graph.add_node(Flatten(), name='flatten', input='pool')

    #######################################################################
    # The fixed-width window (i.e. entire sentence) follows this path
    # through the graph.
    #######################################################################

    if hasattr(args, 'embedding_weights') and args.embedding_weights is not None:
        W = np.load(args.embedding_weights)
        # TODO: second argument to Embedding should be automatically
        # derived from shape of `W`.
        if args.train_embeddings:
            fixed_embedding = Embedding(args.n_vocab, args.n_word_dims,
                    weights=[W],
                    W_constraint=maxnorm(args.embedding_max_norm))
        else:
            fixed_embedding = ImmutableEmbedding(args.n_vocab, args.n_word_dims,
                weights=[W])
    else:
        fixed_embedding = Embedding(args.n_vocab, args.n_word_dims,
            W_constraint=maxnorm(args.embedding_max_norm))

    graph.add_node(fixed_embedding,
            name='fixed-embedding',
            input=args.fixed_window_name)

    graph.add_node(
            TimeDistributedMerge(
                mode=args.time_distributed_merge_mode),
            name='fixed-merge',
            input='fixed-embedding')

    mlp_l1 = Dense(args.n_word_dims, 2*args.n_word_dims, 
            activation='tanh',
            W_regularizer=l2(args.l2_penalty))
    graph.add_node(mlp_l1, name='mlp-l1', input='fixed-merge')

    mlp_l2 = Dense(2*args.n_word_dims, args.n_word_dims,
            activation='tanh',
            W_regularizer=l2(args.l2_penalty))
    graph.add_node(mlp_l2, name='mlp-l2', input='mlp-l1')

    '''
    if 'dropout' in args.regularization_layer:
        graph.add_node(Dropout(args.dropout_p_conv))
    if 'normalization' in args.regularization_layer:
        graph.add_node(BatchNormalization((2*dense1_width,)))
    '''
    #######################################################################
    # The X, args.focus_name, and position inputs meet at the output
    # of the embedding+convolution+pooling subgraph and the input to
    # the first fully-connected layer.
    #######################################################################

    # The + 1 at the end is for the position.  We're initially encoding 
    # position as an integer -- it might be better as a one-hot vector.
    dense1_width = args.n_filters + args.n_word_dims
    print('dense1_width', dense1_width)
    graph.add_node(Dense(dense1_width, 2*dense1_width,
                activation='relu',
                W_regularizer=l2(args.l2_penalty)),
            name='dense1',
            inputs=['flatten', 'mlp-l2'],
            concat_axis=1)
    '''
    if 'dropout' in args.regularization_layer:
        graph.add_node(Dropout(args.dropout_p))
    if 'normalization' in args.regularization_layer:
        graph.add_node(BatchNormalization((2*dense1_width,)))
    '''

    graph.add_node(Dense(2*dense1_width, 2*dense1_width,
                activation='relu'),
            name='dense2',
            input='dense1')
    '''
    if 'dropout' in args.regularization_layer:
        graph.add_node(Dropout(args.dropout_p))
    if 'normalization' in args.regularization_layer:
        graph.add_node(BatchNormalization((2*dense1_width,)))
    '''

    graph.add_node(Dense(2*dense1_width, 2*dense1_width,
                activation='relu'),
            name='dense3',
            input='dense2')
    '''
    if 'dropout' in args.regularization_layer:
        graph.add_node(Dropout(args.dropout_p))
    if 'normalization' in args.regularization_layer:
        graph.add_node(BatchNormalization((2*dense1_width,)))
    '''

    graph.add_node(Dense(2*dense1_width, args.n_classes,
                activation='relu',
                W_regularizer=l2(args.l2_penalty)),
            name='softmax',
            input='dense3')
    #if 'normalization' in args.regularization_layer:
    #    graph.add_node(BatchNormalization((args.n_classes,)))

    # Not sure if add_output is necessary if the output
    # is coming from a single node and there's only one
    # cost function.
    graph.add_output(name=args.target_name, input='softmax')

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

    graph.compile(optimizer=optimizer, loss={ args.target_name: args.loss })

    #history = graph.fit({'input1':X_train, 'input2':X2_train, 'output':y_train}, nb_epoch=10)
    #predictions = graph.predict({'input1':X_test, 'input2':X2_test}) # {'output':...}

    return graph
