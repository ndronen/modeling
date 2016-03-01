import sys
import argparse
import numpy

def kvpair(s):
    try:
        k,v = s.split('=')
        if '.' in v:
            try:
                v = float(v)
            except ValueError:
                pass
        else:
            try:
                v = int(v)
            except ValueError:
                pass
        return k,v
    except:
        raise argparse.ArgumentTypeError(
                '--model-cfg arguments must be KEY=VALUE pairs')

def build_chainer():
    parser = build()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
            help='GPU ID (negative value indicates CPU)')
    return parser

def build_keras():
    parser = build()
    return parser

def build_lasagne():
    parser = build()
    parser.add_argument('--progress', action='store_true',
            help='Whether to display a progress for training and validation')
    return parser

def build():
    parser = argparse.ArgumentParser(
            description='Train a model.')
    parser.add_argument('model_dir', metavar="MODEL_DIR", type=str,
            help='The base directory of this model.  Must contain a model.py (model code) and a model.json (hyperparameters).  Model configuration and weights are saved to model_dir/UUID.')
    parser.add_argument('--model-cfg', type=kvpair, nargs='+', default=[],
            help='Model hyper-parameters as KEY=VALUE pairs; overrides parameters in MODEL_DIR/model.json')
    parser.add_argument('--model-dest', type=str, default='',
            help='Directory to which to copy model.py and model.json.  This overrides copying to model_dir/UUID.')
    parser.add_argument(
            '--mode', type=str, 
            choices=['transient', 'persistent', 'persistent-background'],
            default='persistent',
            help='How to run the model; in "transient" mode, output goes to the console and the model is not saved; in "persistent" mode, output goes to the console and the model is saved; in "persistent-background" mode, output goes to the model.log file and the model is saved.  The default is "persistent"')

    return parser
