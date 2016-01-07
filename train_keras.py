#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import os, sys, shutil
import logging
import json
import uuid
import json
import itertools 

import numpy as np

import theano
import h5py
import six
from sklearn.metrics import classification_report, fbeta_score, accuracy_score

from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.models

sys.path.append('.')

from modeling.callbacks import (ClassificationReport,
        SingleStepLearningRateSchedule)
from modeling.utils import (count_parameters, callable_print,
        setup_logging, setup_model_dir, save_model_info,
        load_model_data, load_model_json, load_target_data,
        build_model_id, build_model_path,
        ModelConfig)
import modeling.parser

def main(args):
    model_id = build_model_id(args)
    model_path = build_model_path(args, model_id)
    setup_model_dir(args, model_path)
    sys.stdout, sys.stderr = setup_logging(args, model_path)

    x_train, y_train = load_model_data(args.train_file,
            args.data_name, args.target_name)
    x_validation, y_validation = load_model_data(
            args.validation_file,
            args.data_name, args.target_name)

    rng = np.random.RandomState(args.seed)

    if args.n_classes > -1:
        n_classes = args.n_classes
    else:
        n_classes = max(y_train)+1

    n_classes, target_names, class_weight = load_target_data(args, n_classes)

    logging.debug("n_classes {0} min {1} max {2}".format(
        n_classes, min(y_train), max(y_train)))

    y_train_one_hot = np_utils.to_categorical(y_train, n_classes)
    y_validation_one_hot = np_utils.to_categorical(y_validation, n_classes)

    logging.debug("y_train_one_hot " + str(y_train_one_hot.shape))
    logging.debug("x_train " + str(x_train.shape))

    min_vocab_index = np.min(x_train)
    max_vocab_index = np.max(x_train)
    logging.debug("min vocab index {0} max vocab index {1}".format(
        min_vocab_index, max_vocab_index))

    json_cfg = load_model_json(args, x_train, n_classes)

    logging.debug("loading model")

    sys.path.append(args.model_dir)
    import model
    from model import build_model

    if args.subsetting_function:
        subsetter = getattr(model, args.subsetting_function)
    else:
        subsetter = None

    def take_subset(subsetter, path, x, y, y_one_hot, n):
        if subsetter is None:
            return x[0:n], y[0:n], y_one_hot[0:n]
        else:
            mask = subsetter(path)
            idx = np.where(mask)[0]
            idx = idx[0:n]
        return x[idx], y[idx], y_one_hot[idx]

    x_train, y_train, y_train_one_hot = take_subset(
            subsetter, args.train_file,
            x_train, y_train, y_train_one_hot,
            n=args.n_train)

    x_validation, y_validation, y_validation_one_hot = take_subset(
            subsetter, args.validation_file,
            x_validation, y_validation, y_validation_one_hot,
            n=args.n_validation)

    logging.debug("y_train_one_hot " + str(y_train_one_hot.shape))
    logging.debug("x_train " + str(x_train.shape))

    model_cfg = ModelConfig(**json_cfg)
    logging.info("model_cfg " + str(model_cfg))
    model = build_model(model_cfg)
    setattr(model, 'stop_training', False)

    logging.info('model has {n_params} parameters'.format(
        n_params=count_parameters(model)))

    if len(args.extra_train_file) > 1:
        callbacks = keras.callbacks.CallbackList()
    else:
        callbacks = []

    save_model_info(args, model_path, model_cfg)

    if not args.no_save:
        callbacks.append(ModelCheckpoint(
            filepath=model_path + '/model-{epoch:04d}.h5',
            verbose=1,
            save_best_only=True))

    callback_logger = logging.info if args.log else callable_print

    if args.n_epochs < sys.maxsize:
        # Number of epochs overrides patience.  If the number of epochs
        # is specified on the command line, the model is trained for
        # exactly that number; otherwise, the model is trained with
        # early stopping using the patience specified in the model 
        # configuration.
        callbacks.append(EarlyStopping(
            monitor='val_loss', patience=model_cfg.patience, verbose=1))

    if args.classification_report:
        cr = ClassificationReport(x_validation, y_validation,
                callback_logger,
                target_names=target_names)
        callbacks.append(cr)

    if model_cfg.optimizer == 'SGD':
        callbacks.append(SingleStepLearningRateSchedule(patience=10))

    if len(args.extra_train_file) > 1:
        args.extra_train_file.append(args.train_file)
        logging.info("Using the following files for training: " +
                ','.join(args.extra_train_file))

        train_file_iter = itertools.cycle(args.extra_train_file)
        current_train = args.train_file

        callbacks._set_model(model)
        callbacks.on_train_begin(logs={})

        epoch = batch = 0

        while True:
            iteration = batch % len(args.extra_train_file)

            logging.info("epoch {epoch} iteration {iteration} - training with {train_file}".format(
                    epoch=epoch, iteration=iteration, train_file=current_train))
            callbacks.on_epoch_begin(epoch, logs={})

            n_train = x_train.shape[0]

            callbacks.on_batch_begin(batch, logs={'size': n_train})

            index_array = np.arange(n_train)
            if args.shuffle:
                rng.shuffle(index_array)

            batches = keras.models.make_batches(n_train, model_cfg.batch_size)
            logging.info("epoch {epoch} iteration {iteration} - starting {n_batches} batches".format(
                    epoch=epoch, iteration=iteration, n_batches=len(batches)))

            avg_train_loss = avg_train_accuracy = 0.
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]

                if isinstance(model, keras.models.Graph):
                    data = {
                            'input': x_train[batch_ids],
                            'output': y_train_one_hot[batch_ids]
                            }
                    train_loss = model.train_on_batch(data, class_weight=class_weight)
                    train_accuracy = 0.
                else:
                    train_loss, train_accuracy = model.train_on_batch(
                            x_train[batch_ids], y_train_one_hot[batch_ids],
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

            batch += 1

            # Validation frequency (this if-block) doesn't necessarily
            # occur in the same iteration as beginning of an epoch
            # (next if-block), so model.evaluate appears twice here.
            if (iteration + 1) % args.validation_freq == 0:
                val_loss, val_acc = model.evaluate(
                        x_validation, y_validation_one_hot,
                        show_accuracy=True,
                        verbose=0 if args.log else 1)
                logging.info("epoch {epoch} iteration {iteration} - val_loss: {val_loss} - val_acc: {val_acc}".format(
                        epoch=epoch, iteration=iteration, val_loss=val_loss, val_acc=val_acc))
                epoch_end_logs = {'iteration': iteration, 'val_loss': val_loss, 'val_acc': val_acc}
                callbacks.on_epoch_end(epoch, epoch_end_logs)

            if batch % len(args.extra_train_file) == 0:
                val_loss, val_acc = model.evaluate(
                        x_validation, y_validation_one_hot,
                        show_accuracy=True,
                        verbose=0 if args.log else 1)
                logging.info("epoch {epoch} iteration {iteration} - val_loss: {val_loss} - val_acc: {val_acc}".format(
                        epoch=epoch, iteration=iteration, val_loss=val_loss, val_acc=val_acc))
                epoch_end_logs = {'iteration': iteration, 'val_loss': val_loss, 'val_acc': val_acc}
                epoch += 1
                callbacks.on_epoch_end(epoch, epoch_end_logs)

            if model.stop_training:
                logging.info("epoch {epoch} iteration {iteration} - done training".format(
                    epoch=epoch, iteration=iteration))
                break

            current_train = next(train_file_iter)
            x_train, y_train = load_model_data(current_train,
                    args.data_name, args.target_name)
            y_train_one_hot = np_utils.to_categorical(y_train, n_classes)

            if epoch > args.n_epochs:
                break

        callbacks.on_train_end(logs={})
    else:
        if isinstance(model, keras.models.Graph):
            data = {
                    'input': x_train,
                    'output': y_train_one_hot
                    }
            validation_data = {
                    'input': x_validation,
                    'output': y_validation_one_hot
                    }
            model.fit(data,
                shuffle=args.shuffle,
                nb_epoch=args.n_epochs,
                batch_size=model_cfg.batch_size,
                #show_accuracy=True,
                validation_data=validation_data,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=2 if args.log else 1)
            y_hat = model.predict(validation_data)
            print('val_acc %.04f' % 
                    accuracy_score(y_validation, np.argmax(y_hat['output'], axis=1)))
        else:
            model.fit(x_train, y_train_one_hot,
                shuffle=args.shuffle,
                nb_epoch=args.n_epochs,
                batch_size=model_cfg.batch_size,
                show_accuracy=True,
                validation_data=(x_validation, y_validation_one_hot),
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=2 if args.log else 1)

if __name__ == '__main__':
    parser = modeling.parser.build_keras()
    sys.exit(main(parser.parse_args()))
