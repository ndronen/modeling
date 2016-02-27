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
from sklearn.metrics import accuracy_score

from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.models

sys.path.append('.')

from modeling.callbacks import (ClassificationReport,
        ConfusionMatrix, PredictionCallback,
        DelegatingMetricCallback,
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

    rng = np.random.RandomState(args.seed)

    json_cfg = load_model_json(args, x_train=None, n_classes=None)
    model_cfg = ModelConfig(**json_cfg)
    if args.verbose:
        print("model_cfg " + str(model_cfg))

    sys.path.append(args.model_dir)
    import model
    from model import build_model, fit_model, load_train, load_validation

    train_data = load_train(args, model_cfg)
    validation_data = load_validation(args, model_cfg)

    if args.verbose:
        print("loading model")
    model = build_model(model_cfg, train_data, validation_data)
    fit_model(model, train_data, validation_data, args)

if __name__ == '__main__':
    parser = modeling.parser.build_keras()
    sys.exit(main(parser.parse_args()))
