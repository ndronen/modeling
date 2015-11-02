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

from modeling.lasagne_model import Classifier
from modeling.utils import (
        load_model_data, load_model_json, build_model_id, build_model_path,
        setup_model_dir, setup_logging, ModelConfig)
import modeling.parser

def main(args):
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

    print('y_train', y_train.tolist())

    rng = np.random.RandomState(args.seed)

    N = len(x_train)
    N_validation = len(x_validation)

    n_classes = max(np.unique(y_train)) + 1
    json_cfg = load_model_json(args, x_train, n_classes)

    print('args.model_dir', args.model_dir)
    sys.path.append(args.model_dir)
    from model import Model
    model_cfg = ModelConfig(**json_cfg)
    print('model_cfg', model_cfg)
    model = Model(model_cfg)
    setattr(model, 'stop_training', False)
    
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
    
        '''
        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=N).start()
        '''

        train_loss = 0

        for j, i in enumerate(six.moves.range(0, N, model_cfg.batch_size)):
            #pbar.update(j+1)
            x_batch = x_train[perm[i:i + model_cfg.batch_size]]
            y_batch = y_train[perm[i:i + model_cfg.batch_size]]
            if len(x_batch) != model_cfg.batch_size:
                # TODO: how do other frameworks solve this?
                continue
            train_loss += model.fit(x_batch, y_batch)

        #pbar.finish()
        print('train epoch={}, mean loss={}'.format(epoch, train_loss/float(N)))
    
        # Validation set evaluation
        val_accuracy = 0.
        val_loss = 0.

        pbar = progressbar.ProgressBar(term_width=40,
            widgets=[' ', progressbar.Percentage(),
            ' ', progressbar.ETA()],
            maxval=N_validation).start()

        for i in six.moves.range(0, N_validation, model_cfg.batch_size):
            pbar.update(i+1)
            x_batch = x_validation[i:i + model_cfg.batch_size]
            y_batch = y_validation[i:i + model_cfg.batch_size]
            loss, acc = model.evaluate(x_batch, y_batch)
            val_loss += loss.item()
            val_accuracy += acc.item()

        pbar.finish()
        val_loss /= float(N_validation)
        val_accuracy /= float(N_validation)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            if model_path is not None:
                model.save_weights(model_path + '.npz')
                cPickle.dump(model, open(model_path + '.pkl', 'w'))

        print('validation epoch={}, mean loss={}, accuracy={} best=[accuracy={} epoch={}]'.format(
            epoch, val_loss, val_accuracy, 
            best_accuracy,
            best_epoch))
    
        epoch += 1
    
if __name__ == '__main__':
    parser = modeling.parser.build_lasagne()
    sys.exit(main(parser.parse_args()))
