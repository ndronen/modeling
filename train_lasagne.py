#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import sys
import six
import argparse
import progressbar
import copy
import cPickle
import itertools

import numpy as np
import pandas as pd

from modeling.lasagne_model import Classifier
from modeling.utils import (
        load_model_data, load_model_json, build_model_id, build_model_path,
        setup_model_dir, setup_logging, ModelConfig)
import modeling.parser

def keep_training(epoch, best_epoch, model_cfg):
    if model_cfg.n_epochs is not None and epoch > model_cfg.n_epochs:
            return False
    if epoch > 1 and epoch - best_epoch > model_cfg.patience:
        return False
    return True

def train_one_epoch(model, x_train, y_train, args, model_cfg, progress=False):
    n = len(x_train)

    if args.shuffle:
        perm = np.random.permutation(n)
    else:
        perm = np.arange(n)

    if progress:
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=n).start()
    else:
        pbar = None

    train_loss = 0

    for j, i in enumerate(six.moves.range(0, n, model_cfg.batch_size)):
        if progress:
            pbar.update(j+1)
        x = x_train[perm[i:i + model_cfg.batch_size]]
        y = y_train[perm[i:i + model_cfg.batch_size]]
        if len(x) != model_cfg.batch_size:
            # TODO: how do other frameworks solve this?
            continue
        train_loss += model.fit(x, y)

    if progress:
        pbar.finish()

    return train_loss/float(n)

def validate(model, x_valid, y_valid, args, model_cfg, progress=False):
    n = len(x_valid)

    if progress:
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=n).start()
    else:
        pbar = None

    val_accuracy = 0.
    val_loss = 0.

    for i in six.moves.range(0, n, model_cfg.batch_size):
        if progress:
            pbar.update(i+1)
        x = x_valid[i:i + model_cfg.batch_size]
        y = y_valid[i:i + model_cfg.batch_size]
        loss, acc = model.evaluate(x, y)
        val_loss += loss
        val_accuracy += acc

    if progress:
        pbar.finish()

    return val_loss/float(n), val_accuracy/float(n)

def main(args):
    model_id = build_model_id(args)
    model_path = build_model_path(args, model_id)
    setup_model_dir(args, model_path)
    sys.stdout, sys.stderr = setup_logging(args)

    rng = np.random.RandomState(args.seed)

    x_train, y_train = load_model_data(args.train_file,
            args.data_name, args.target_name,
            n=args.n_train)

    x_valid, y_valid = load_model_data(
            args.validation_file,
            args.data_name, args.target_name,
            n=args.n_validation)

    train_files = args.extra_train_file + [args.train_file]
    train_files_iter = itertools.cycle(train_files)

    n_classes = max(np.unique(y_train)) + 1
    json_cfg = load_model_json(args, x_train, n_classes)

    sys.path.append(args.model_dir)
    from model import Model
    model_cfg = ModelConfig(**json_cfg)
    model = Model(model_cfg)
    setattr(model, 'stop_training', False)
    
    best_accuracy = 0.
    best_epoch = 0
    
    epoch = 1
    iteration = 0
    
    while True:
        if not keep_training(epoch, best_epoch, model_cfg):
            break

        train_loss = train_one_epoch(model, x_train, y_train,
                args, model_cfg, progress=args.progress)

        val_loss, val_accuracy = validate(model, x_valid, y_valid,
                args, model_cfg, progress=args.progress)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            if model_path is not None:
                model.save_weights(model_path + '.npz')
                cPickle.dump(model, open(model_path + '.pkl', 'w'))

        print('epoch={epoch:05d}, iteration={iteration:05d}, loss={loss:.04f}, val_loss={val_loss:.04f}, val_acc={val_acc:.04f} best=[accuracy={best_accuracy:.04f} epoch={best_epoch:05d}]'.format(
            epoch=epoch, iteration=iteration,
            loss=train_loss, val_loss=val_loss, val_acc=val_accuracy, 
            best_accuracy=best_accuracy, best_epoch=best_epoch))
    
        iteration += 1
        if iteration % len(train_files) == 0:
            epoch += 1

        x_train, y_train = load_model_data(
                next(train_files_iter),
                args.data_name, args.target_name,
                n=args.n_train)
    
if __name__ == '__main__':
    parser = modeling.parser.build_lasagne()
    sys.exit(main(parser.parse_args()))
