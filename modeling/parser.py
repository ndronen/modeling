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
    parser.add_argument('--graph-marshalling-class', type=str, default="GraphMarshaller",
            help='Name of class in model.py to use to marshall data for keras graphs')
    parser.add_argument('--early-stopping-metric', type=str, default='val_loss',
            choices=['val_loss', 'val_f1', 'val_f2', 'val_f0.5'],
            help='Metric to use for early stopping')
    parser.add_argument('--checkpoint-metric', type=str, default='val_loss',
            choices=['val_loss', 'val_f1', 'val_f2', 'val_f0.5'],
            help='Metric to use for model checkpointing')
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
    parser.add_argument('train_file', metavar='TRAIN_FILE', type=str,
            help='HDF5 file of training examples.')
    parser.add_argument('validation_file', metavar='VALIDATION_FILE', type=str,
            help='HDF5 file of validation examples.')
    parser.add_argument('data_name', nargs='+', type=str,
            help='Name(s) of the data variable(s) in input HDF5 file.')
    parser.add_argument('--target-name', default='target_code', type=str,
            help='Name of the target variable in input HDF5 file.')
    parser.add_argument('--extra-train-file', type=str, nargs='+', default=[],
            help='path to one or more extra train files, useful for when training set is too big to fit into memory.')
    parser.add_argument('--model-cfg', type=kvpair, nargs='+', default=[],
            help='Model hyper-parameters as KEY=VALUE pairs; overrides parameters in MODEL_DIR/model.json')
    parser.add_argument('--model-dest', type=str, default='',
            help='Directory to which to copy model.py and model.json.  This overrides copying to model_dir/UUID.')
    parser.add_argument('--target-data', type=str,
            help='Pickled dictionary of target data from sklearn.preprocessing.LabelEncoder.  The dictionary must contain a key `TARGET_NAME` that maps either to a list of target names or a dictionary mapping target names to their class weights (useful for imbalanced data sets')
    #parser.add_argument('--use-class-weights', action='store_true',
    #        help='Whether to use the class weights from the target data file during training (see --target-data)')
    parser.add_argument('--description', type=str,
            help='Short description of this model (data, hyperparameters, etc.)')
    parser.add_argument('--shuffle', default=False, action='store_true',
            help='Shuffle the data in each minibatch')
    parser.add_argument('--n-epochs', default=sys.maxsize, type=int,
            help='The maximum number of epochs to train')
    parser.add_argument('--early-stopping', action='store_true',
            help='Whether to use early stopping by monitoring validation set loss')
    parser.add_argument('--n-train', default=sys.maxsize, type=int,
            help='The number of training examples to use')
    parser.add_argument('--n-validation', default=sys.maxsize, type=int,
            help='The number of validation examples to use')
    parser.add_argument('--n-embeddings', default=-1, type=int,
            help="The number of words in the model's vocabulary")
    parser.add_argument('--n-classes', default=-1, type=int,
            help='The number of classes in TARGET_NAME')
    parser.add_argument('--class-weight-auto', action='store_true',
            help='Set the class weights based on their frequency in the training set')
    parser.add_argument('--class-weight-exponent', type=float,
            help='Raise the class weights to the given power')
    parser.add_argument('--log', action='store_true',
            help='Whether to send console output to log file')
    parser.add_argument('--verbose', action='store_true',
            help='Whether to print more during initialization')
    parser.add_argument('--no-save', action='store_true',
            help='Disable saving/copying of model.py and model.json to a unique directory for reproducibility')
    parser.add_argument('--classification-report', action='store_true',
            help='Include an sklearn classification report on the validation set at end of each epoch')
    parser.add_argument('--confusion-matrix', action='store_true',
            help='Include an sklearn confusion matrix on the validation set at end of each epoch')
    parser.add_argument('--validation-freq', default=1, type=int,
            help='How often to run validation set (only relevant with --extra-train-file')
    parser.add_argument('--subsetting-function', type=str,
            help='Name of function in model.py to use to take subsets of training and validation data')
    parser.add_argument('--preprocessing-class', type=str,
            help='Name of class in model.py to use to preprocess training and validation data')
    parser.add_argument('--save-all-checkpoints', action='store_true',
            help='Save the weights of the model in separate files at every checkpoint (i.e. every epoch if --save-every-epoch; otherwise, at every new best validation set performance).  This causes files to be stored as model-XXXX.h5, where XXXX is the epoch number; without this option, the model is stored as model.h5')
    parser.add_argument('--save-every-epoch', action='store_true',
            help='Save the weights of the model every epoch; see also --save-all-checkpoints')
    parser.add_argument('--seed', default=17, type=int,
            help='The seed for the random number generator')

    return parser
