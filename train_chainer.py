#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import sys
import six
import argparse
import progressbar
import copy
import cPickle

import numpy as np
import pandas as pd

import chainer
from chainer import cuda
from modeling.chainer_model import Classifier
from modeling.utils import (
        load_model_data, load_model_json, build_model_id, build_model_path,
        setup_model_dir, setup_logging, ModelConfig)
import modeling.parser

def main(args):
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    model_id = build_model_id(args)
    model_path = build_model_path(args, model_id)
    setup_model_dir(args, model_path)
    sys.stdout, sys.stderr = setup_logging(args)

    x_train, y_train = load_model_data(args.train_file,
            args.data_name, args.target_name,
            n=args.n_train)
    x_validation, y_validation = load_model_data(
            args.validation_file,
            args.data_name, args.target_name,
            n=args.n_validation)

    rng = np.random.RandomState(args.seed)

    N = len(x_train)
    N_validation = len(x_validation)

    n_classes = max(np.unique(y_train)) + 1
    json_cfg = load_model_json(args, x_train, n_classes)

    print('args.model_dir', args.model_dir)
    sys.path.append(args.model_dir)
    from model import Model
    model_cfg = ModelConfig(**json_cfg)
    model = Model(model_cfg)
    setattr(model, 'stop_training', False)
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    
    best_accuracy = 0.
    best_epoch = 0
    
    def keep_training(epoch, best_epoch):
        if model_cfg.n_epochs is not None and epoch > model_cfg.n_epochs:
                return False
        if epoch > 1 and epoch - best_epoch > model_cfg.patience:
            return False
        return True
    
    epoch = 1
    
    while True:
        if not keep_training(epoch, best_epoch):
            break
    
        if args.shuffle:
            perm = np.random.permutation(N)
        else:
            perm = np.arange(N)
    
        sum_accuracy = 0
        sum_loss = 0

        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=N).start()

        for j, i in enumerate(six.moves.range(0, N, model_cfg.batch_size)):
            pbar.update(j+1)
            x_batch = xp.asarray(x_train[perm[i:i + model_cfg.batch_size]].flatten())
            y_batch = xp.asarray(y_train[perm[i:i + model_cfg.batch_size]])
            pred, loss, acc = model.fit(x_batch, y_batch)
            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)

        pbar.finish()
        print('train epoch={}, mean loss={}, accuracy={}'.format(
            epoch, sum_loss / N, sum_accuracy / N))
    
        # Validation set evaluation
        sum_accuracy = 0
        sum_loss = 0

        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=N_validation).start()

        for i in six.moves.range(0, N_validation, model_cfg.batch_size):
            pbar.update(i+1)
            x_batch = xp.asarray(x_validation[i:i + model_cfg.batch_size].flatten())
            y_batch = xp.asarray(y_validation[i:i + model_cfg.batch_size])
            pred, loss, acc = model.predict(x_batch, target=y_batch)
            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)

        pbar.finish()
        validation_accuracy = sum_accuracy / N_validation
        validation_loss = sum_loss / N_validation
    
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            if model_path is not None:
                if args.gpu >= 0:
                    model.to_cpu()
                store = {
                        'args': args,
                        'model': model,
                    }
                cPickle.dump(store, open(model_path + '.store', 'w'))
                if args.gpu >= 0:
                    model.to_gpu()
    
        print('validation epoch={}, mean loss={}, accuracy={} best=[accuracy={} epoch={}]'.format(
            epoch, validation_loss, validation_accuracy, 
            best_accuracy,
            best_epoch))
    
        epoch += 1
    
if __name__ == '__main__':
    parser = modeling.parser.build_chainer()
    sys.exit(main(parser.parse_args()))
