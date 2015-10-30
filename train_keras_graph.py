#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import os, sys, shutil
import argparse
import logging
import json
import uuid
import json
import itertools 

import numpy as np

import theano
import h5py
import six
from sklearn.metrics import classification_report, fbeta_score

from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping

sys.path.append('.')

from modeling.callbacks import ClassificationReport
from modeling.utils import (count_parameters, callable_print,
        ModelConfig, LoggerWriter)
import modeling.parser

def main(args):
    if args.graph_dest:
        graph_id = args.graph_dest
    else:
        graph_id = uuid.uuid1().hex

    graph_path = args.graph_dir + '/' + graph_id + '/'

    if not args.no_save:
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)
        print("graph path is " + graph_path)

    if args.log and not args.no_save:
        logging.basicConfig(filename=graph_path + 'graph.log',
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M',
                level=logging.DEBUG)
        sys.stdout = LoggerWriter(logging.info)
        sys.stderr = LoggerWriter(logging.warning)
    else:
        logging.basicConfig(level=logging.DEBUG)

    variable_window_train, target_train = load_model_data(
            args.train_file,
            args.variable_window_name,
            args.target_name)

    variable_window_valid, target_valid = load_model_data(
            args.valid_file,
            args.variable_window_name, 
            args.target_name)

    '''
    if len(target_train) > args.n_train:
        logging.info("Reducing training set size from " +
                str(len(target_train)) + " to " + str(args.n_train))
        target_train = target_train[0:args.n_train]
        window_train = window_train[0:args.n_train, :]

    if len(target_valid) > args.n_valid:
        logging.info("Reducing valid set size from " +
                str(len(target_valid)) + " to " + str(args.n_valid))
        target_valid = target_valid[0:args.n_valid]
        window_valid = window_valid[0:args.n_valid, :]
    '''
    
    rng = np.random.RandomState(args.seed)

    if args.target_data:
        target_names_dict = json.load(open(args.target_data))

        try:
            target_data = target_names_dict[args.target_name]
        except KeyError:
            raise ValueError("Invalid key " + args.target_name +
                    " for dictionary in " + args.target_data)

        if isinstance(target_data, dict):
            try:
                target_names = target_data['names']
                class_weight = target_data['weights']
            except KeyError, e:
                raise ValueError("Target data dictionary from " +
                        args.target_data + "is missing a key: " + str(e))
        elif isinstance(target_data, list):
            target_names = target_data
            class_weight = None
        else:
            raise ValueError("Target data must be list or dict, not " +
                    str(type(target_data)))

        n_classes = len(target_names)
    else:
        target_names = None
        class_weight = None
        if args.n_classes > -1:
            n_classes = args.n_classes
        else:
            n_classes = max(target_train) + 1

    if class_weight is not None:
        # Keys are strings in JSON; convert them to int.
        for k,v in class_weight.iteritems():
            del class_weight[k]
            class_weight[int(k)] = v

    logging.debug("n_classes {0} min {1} max {2}".format(
        n_classes, min(target_train), max(target_train)))

    target_train_one_hot = np_utils.to_categorical(target_train, n_classes)
    target_valid_one_hot = np_utils.to_categorical(target_valid, n_classes)

    logging.debug("target_train_one_hot " + str(target_train_one_hot.shape))
    logging.debug("variable_window_train " + str(variable_window_train.shape))

    variable_input_width = variable_window_train.shape[1]

    min_vocab_index = np.min(variable_window_train)
    max_vocab_index = np.max(variable_window_train)
    logging.debug("min vocab index {0} max vocab index {1}".format(
        min_vocab_index, max_vocab_index))

    json_cfg = json.load(open(args.graph_dir + '/graph.json'))

    # Copy graph parameters provided on the command-line.
    for k,v in args.graph_cfg:
        json_cfg[k] = v

    # Add a few values to the dictionary that are properties
    # of the training data.
    json_cfg['train_file'] = args.train_file
    json_cfg['valid_file'] = args.valid_file
    json_cfg['n_vocab'] = max(args.n_vocab, np.max(variable_window_train) + 1)
    json_cfg['variable_input_width'] = variable_window_train.shape[1]
    json_cfg['n_classes'] = n_classes
    json_cfg['seed'] = args.seed
    json_cfg['variable_window_name'] = args.variable_window_name
    json_cfg['target_name'] = args.target_name

    logging.debug("loading graph")

    sys.path.append(args.graph_dir)
    from graph import build_graph
    graph_cfg = ModelConfig(**json_cfg)
    graph = build_graph(graph_cfg)

    setattr(graph, 'stop_training', False)

    logging.info('graph has {n_params} parameters'.format(
        n_params=count_parameters(graph)))

    if args.extra_train_file is not None:
        callbacks = keras.callbacks.CallbackList()
    else:
        callbacks = []

    if not args.no_save:
        if args.description:
            with open(graph_path + '/README.txt', 'w') as f:
                f.write(args.description + '\n')

        # Save graph hyperparameters and code.
        for graph_file in ['graph.py', 'graph.json']:
            shutil.copyfile(args.graph_dir + '/' + graph_file,
                    graph_path + '/' + graph_file)

        json.dump(vars(args), open(graph_path + '/args.json', 'w'))

        # And weights.
        callbacks.append(ModelCheckpoint(
            filepath=graph_path + '/graph.h5',
            verbose=1,
            save_best_only=True))

    callback_logger = logging.info if args.log else callable_print

    callbacks.append(EarlyStopping(
        monitor='val_loss', patience=graph_cfg.patience, verbose=1))

    if args.classification_report:
        cr = ClassificationReport(window_valid, target_valid,
                callback_logger,
                target_names=target_names,
                error_classes_only=args.error_classes_only)
        callbacks.append(cr)

    if args.extra_train_file is not None:
        raise ValueError('using multiple train files is not supported yet with graphs')

        args.extra_train_file.append(args.train_file)
        logging.info("Using the following files for training: " +
                ','.join(args.extra_train_file))

        train_file_iter = itertools.cycle(args.extra_train_file)
        current_train = args.train_file

        callbacks._set_graph(graph)
        callbacks.on_train_begin(logs={})

        epoch = batch = 0

        while True:
            iteration = batch % len(args.extra_train_file)

            logging.info("epoch {epoch} iteration {iteration} - training with {train_file}".format(
                    epoch=epoch, iteration=iteration, train_file=current_train))
            callbacks.on_epoch_begin(epoch, logs={})

            n_train = window_train.shape[0]

            callbacks.on_batch_begin(batch, logs={'size': n_train})

            index_array = np.arange(n_train)
            if args.shuffle:
                index_array = rng.shuffle(index_array)

            batches = keras.graphs.make_batches(n_train, graph_cfg.batch_size)
            logging.info("epoch {epoch} iteration {iteration} - starting {n_batches} batches".format(
                    epoch=epoch, iteration=iteration, n_batches=len(batches)))

            avg_train_loss = avg_train_accuracy = 0.
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]

                train_loss, train_accuracy = graph.train_on_batch(
                        window_train[batch_ids], target_train_one_hot[batch_ids],
                        accuracy=True, class_weight=class_weight)

                batch_end_logs = {'loss': train_loss, 'accuracy': train_accuracy}

                avg_train_loss = (avg_train_loss * batch_index + train_loss)/(batch_index + 1)
                avg_train_accuracy = (avg_train_accuracy * batch_index + train_accuracy)/(batch_index + 1)

                callbacks.on_batch_end(batch,
                        logs={'loss': train_loss, 'accuracy': train_accuracy})

            logging.info("epoch {epoch} iteration {iteration} - finished {n_batches} batches".format(
                    epoch=epoch, iteration=iteration, n_batches=len(batches)))

            logging.info("epoch {epoch} iteration {iteration} - loss: {loss} - acc: {acc}".format(
                    epoch=epoch, iteration=iteration, loss=avg_train_loss, acc=avg_train_accuracy))

            val_loss, val_acc = graph.evaluate(
                    window_valid, target_valid_one_hot,
                    show_accuracy=True,
                    verbose=0 if args.log else 1)

            logging.info("epoch {epoch} iteration {iteration} - val_loss: {val_loss} - val_acc: {val_acc}".format(
                    epoch=epoch, iteration=iteration, val_loss=val_loss, val_acc=val_acc))
            epoch_end_logs = {'iteration': iteration, 'val_loss': val_loss, 'val_acc': val_acc}
            callbacks.on_epoch_end(epoch, epoch_end_logs)

            if graph.stop_training:
                logging.info("epoch {epoch} iteration {iteration} - done training".format(
                    epoch=epoch, iteration=iteration))
                break

            current_train = next(train_file_iter)
            window_train, target_train = load_model_data(
                    current_train,
                    args.variable_window_name,
                    args.target_name)
            target_train_one_hot = np_utils.to_categorical(target_train, n_classes)

            batch += 1
            if batch % len(args.extra_train_file) == 0:
                epoch += 1

            if epoch > args.n_epochs:
                break

        callbacks.on_train_end(logs={})
    else:
        train_data = {
                args.variable_window_name: variable_window_train,
                args.target_name: target_train_one_hot
                }
        print([(k, train_data[k].shape) for k in train_data.keys()])

        valid_data = {
                args.variable_window_name: variable_window_valid,
                args.target_name: target_valid_one_hot
                }
        print([(k, valid_data[k].shape) for k in valid_data.keys()])

        print('inputs')
        print(graph.inputs)
        for k,v in graph.inputs.iteritems():
            x = [k, v]
            if hasattr(v, 'input'):
                x.append(v.input)
            if hasattr(v, 'output'):
                x.append(x.output)
            print(x)

        print('nodes')
        print(graph.nodes)
        for k,v in graph.nodes.iteritems():
            x = [k, v]
            if hasattr(v, 'input'):
                x.append(v.input)
            if hasattr(v, 'output'):
                x.append(x.output)
            print(x)
        print('outputs')
        print(graph.outputs)
        for k,v in graph.outputs.iteritems():
            x = [k, v]
            if hasattr(v, 'input'):
                x.append(v.input)
            if hasattr(v, 'output'):
                x.append(x.output)
            print(x)

        graph.fit(train_data,
            shuffle=args.shuffle,
            nb_epoch=args.n_epochs,
            batch_size=graph_cfg.batch_size,
            #show_accuracy=True,
            #validation_data=(valid_data),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2 if args.log else 1)

if __name__ == '__main__':
    parser = modeling.parser.build_keras()
    sys.exit(main(parser.parse_args()))
