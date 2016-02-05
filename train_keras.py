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
from sklearn.metrics import (accuracy_score,
        f1_score, fbeta_score,
        classification_report, confusion_matrix)

from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.models

sys.path.append('.')

from modeling.callbacks import (ClassificationReport,
        ConfusionMatrix, PredictionCallback,
        EarlyStoppingWithMetric,
        SingleStepLearningRateSchedule)
from modeling.utils import (count_parameters, callable_print,
        setup_logging, setup_model_dir, save_model_info,
        load_model_data, load_model_json, load_target_data,
        build_model_id, build_model_path,
        ModelConfig)
import modeling.preprocess
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

    if len(class_weight) == 0 and args.class_weight_auto:
        n_samples = len(y_train)
        weights = float(n_samples) / (n_classes * np.bincount(y_train))
        if args.class_weight_exponent:
            weights = weights**args.class_weight_exponent
        class_weight = dict(zip(range(n_classes), weights))

    if args.verbose:
        logging.debug("n_classes {0} min {1} max {2}".format(
            n_classes, min(y_train), max(y_train)))

    y_train_one_hot = np_utils.to_categorical(y_train, n_classes)
    y_validation_one_hot = np_utils.to_categorical(y_validation, n_classes)

    if args.verbose:
        logging.debug("y_train_one_hot " + str(y_train_one_hot.shape))
        logging.debug("x_train " + str(x_train.shape))

    min_vocab_index = np.min(x_train)
    max_vocab_index = np.max(x_train)

    if args.verbose:
        logging.debug("min vocab index {0} max vocab index {1}".format(
            min_vocab_index, max_vocab_index))

    json_cfg = load_model_json(args, x_train, n_classes)

    if args.verbose:
        logging.debug("loading model")

    sys.path.append(args.model_dir)
    import model
    from model import build_model

    #######################################################################      
    # Subsetting
    #######################################################################      
    if args.subsetting_function:
        subsetter = getattr(M, args.subsetting_function)
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

    #######################################################################      
    # Preprocessing
    #######################################################################      
    if args.preprocessing_class:
        preprocessor = getattr(M, args.preprocessing_class)(seed=args.seed)
    else:
        preprocessor = modeling.preprocess.NullPreprocessor()

    if args.verbose:
        logging.debug("y_train_one_hot " + str(y_train_one_hot.shape))
        logging.debug("x_train " + str(x_train.shape))

    model_cfg = ModelConfig(**json_cfg)
    if args.verbose:
        logging.info("model_cfg " + str(model_cfg))
    net = build_model(model_cfg)
    setattr(net, 'stop_training', False)

    marshaller = None
    if isinstance(net, keras.models.Graph):
        marshaller = getattr(model, args.graph_marshalling_class)()

    logging.info('model has {n_params} parameters'.format(
        n_params=count_parameters(net)))

    if len(args.extra_train_file) > 1:
        callbacks = keras.callbacks.CallbackList()
    else:
        callbacks = []

    save_model_info(args, model_path, model_cfg)

    if not args.no_save:
        if args.save_all_checkpoints:
            filepath = model_path + '/model-{epoch:04d}.h5'
        else:
            filepath = model_path + '/model.h5'
        callbacks.append(ModelCheckpoint(
            filepath=filepath,
            verbose=1,
            save_best_only=not args.save_every_epoch))

    callback_logger = logging.info if args.log else callable_print

    #######################################################################      
    # Callbacks that need validation set predictions.
    #######################################################################      

    pc = PredictionCallback(x_validation, callback_logger,
            marshaller=marshaller)
    callbacks.append(pc)

    if args.classification_report:
        cr = ClassificationReport(x_validation, y_validation,
                callback_logger,
                target_names=target_names)
        pc.add(cr)
    
    if args.confusion_matrix:
        cm = ConfusionMatrix(x_validation, y_validation,
                callback_logger)
        pc.add(cm)

    if args.early_stopping or args.early_stopping_metric is not None:
        if args.early_stopping_metric is None:
            callbacks.append(EarlyStopping(
                    monitor='val_loss', patience=model_cfg.patience,
                    verbose=1))
        else:
            es = EarlyStopping(monitor='metric',
                    mode='max', patience=model_cfg.patience, verbose=1)
            if args.early_stopping_metric == 'f1':
                metric = f1_score
            elif args.early_stopping_metric == 'f2':
                metric = lambda y,y_hat: fbeta_score(y, y_hat, beta=2)
            elif args.early_stopping_metric == 'f0.5':
                metric = lambda y,y_hat: fbeta_score(y, y_hat, beta=0.5)
            else:
                raise ValueError(("don't know the early stopping metric %s" % 
                        args.early_stopping_metric))

            cb = EarlyStoppingWithMetric(
                    x_validation, y_validation, callback_logger,
                    delegate=es, metric=metric, marshaller=marshaller)
            pc.add(cb)

    if model_cfg.optimizer == 'SGD':
        callbacks.append(SingleStepLearningRateSchedule(patience=10))

    if len(args.extra_train_file) > 1:
        args.extra_train_file.append(args.train_file)
        logging.info("Using the following files for training: " +
                ','.join(args.extra_train_file))

        train_file_iter = itertools.cycle(args.extra_train_file)
        current_train = args.train_file

        callbacks._set_model(net)
        callbacks.on_train_begin(logs={})

        epoch = batch = 0

        while True:
            x_train, y_train_one_hot = preprocessor.fit_transform(
                    x_train, y_train_one_hot)
            x_validation, y_validation_one_hot = preprocessor.transform(
                    x_validation, y_validation_one_hot)

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

                if isinstance(net, keras.models.Graph):
                    train_data = marshaller.marshal(
                            x_train[batch_ids], y_train_one_hot[batch_ids])
                    train_loss = net.train_on_batch(
                            train_data, class_weight=class_weight)
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
            kwargs = { 'verbose': 0 if args.log else 1 }
            pargs = []
            validation_data = {}
            if isinstance(net, keras.models.Graph):
                validation_data = marshaller.marshal(
                        x_validation, y_validation_one_hot)
                pargs = [validation_data]
            else:
                pargs = [x_validation, y_validation_one_hot]
                kwargs['show_accuracy'] = True

            if (iteration + 1) % args.validation_freq == 0:
                if isinstance(net, keras.models.Graph):
                    val_loss = net.evaluate(*pargs, **kwargs)
                    y_hat = net.predict(validation_data)
                    val_acc = accuracy_score(y_validation, np.argmax(y_hat['output'], axis=1))
                else:
                    val_loss, val_acc = net.evaluate(
                            *pargs, **kwargs)
                logging.info("epoch {epoch} iteration {iteration} - val_loss: {val_loss} - val_acc: {val_acc}".format(
                        epoch=epoch, iteration=iteration, val_loss=val_loss, val_acc=val_acc))
                epoch_end_logs = {'iteration': iteration, 'val_loss': val_loss, 'val_acc': val_acc}
                callbacks.on_epoch_end(epoch, epoch_end_logs)

            if batch % len(args.extra_train_file) == 0:
                if isinstance(net, keras.models.Graph):
                    val_loss = net.evaluate(*pargs, **kwargs)
                    y_hat = net.predict(validation_data)
                    val_acc = accuracy_score(y_validation, np.argmax(y_hat['output'], axis=1))
                else:
                    val_loss, val_acc = net.evaluate(
                            *pargs, **kwargs)
                logging.info("epoch {epoch} iteration {iteration} - val_loss: {val_loss} - val_acc: {val_acc}".format(
                        epoch=epoch, iteration=iteration, val_loss=val_loss, val_acc=val_acc))
                epoch_end_logs = {'iteration': iteration, 'val_loss': val_loss, 'val_acc': val_acc}
                epoch += 1
                callbacks.on_epoch_end(epoch, epoch_end_logs)

            if net.stop_training:
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
        x_train, y_train_one_hot = preprocessor.fit_transform(
                x_train, y_train_one_hot)
        x_validation, y_validation_one_hot = preprocessor.transform(
                x_validation, y_validation_one_hot)

        if isinstance(net, keras.models.Graph):
            train_data = marshaller.marshal(
                    x_train, y_train_one_hot)
            validation_data = marshaller.marshal(
                    x_validation, y_validation_one_hot)
            net.fit(train_data,
                shuffle=args.shuffle,
                nb_epoch=args.n_epochs,
                batch_size=model_cfg.batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=2 if args.log else 1)
        else:
            net.fit(x_train, y_train_one_hot,
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
