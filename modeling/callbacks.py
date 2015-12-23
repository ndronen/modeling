import os
import numpy as np
from keras.callbacks import Callback
import keras.callbacks
import numpy as np
import six
from sklearn.metrics import classification_report, fbeta_score

'''
class SklearnMetricCheckpointClassification(Callback):
    def __init__(self, model_path, x, y, metric='f1_score', verbose=0, save_best_only=False):
        super(Callback, self).__init__()
        self.__dict__.update(locals())
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        if self.save_best_only:
            self.model.predict_classses

            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Can save best model only with %s available, skipping." % (self.monitor), RuntimeWarning)
            else:
                if current < self.best:
                    if self.verbose > 0:
                        print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                              % (epoch, self.monitor, self.best, current, self.filepath))
                    self.best = current
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, self.filepath))
            self.model.save_weights(self.filepath, overwrite=True)
'''

class ClassificationReport(Callback):
    def __init__(self, x, y, logger, target_names=None, iteration_freq=10):
        self.x = x
        self.y = y
        self.logger = logger
        self.iteration_freq = iteration_freq

        if target_names is not None:
            labels = np.arange(len(target_names))
        else:
            labels = None

        self.labels = labels
        self.target_names = [str(tn) for tn in target_names]

    def on_epoch_end(self, epoch, logs={}):
        if 'iteration' in logs.keys() and logs['iteration'] % self.iteration_freq != 0:
            # If we've broken a large training set into smaller chunks, we don't
            # need to run the classification report after every chunk.
            return

        y_hat = self.model.predict_classes(self.x, verbose=0)
        fbeta = fbeta_score(self.y, y_hat, beta=0.5, average='weighted')
        report = classification_report(
                self.y, y_hat,
                labels=self.labels, target_names=self.target_names)

        if 'iteration' in logs.keys():
            self.logger("epoch {epoch} iteration {iteration} - val_fbeta(beta=0.5): {fbeta}".format(
                epoch=epoch, iteration=logs['iteration'], fbeta=fbeta))
        else:
            self.logger("epoch {epoch} - val_fbeta(beta=0.5): {fbeta}".format(
                epoch=epoch, fbeta=fbeta))

        self.logger(report)

class OptimizerMonitor(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self.model.optimizer, 'lr'):
            return

        lr = self.model.optimizer.lr.get_value()
        optimizer_state = str({ 'lr': lr })

        if 'iteration' in logs.keys():
            self.logger("epoch {epoch} iteration {iteration} - optimizer state {optimizer_state}".format(
                epoch=epoch, iteration=logs['iteration'], optimizer_state=optimizer_state))
        else:
            self.logger("epoch {epoch} - optimizer state {optimizer_state}".format(
                epoch=epoch, optimizer_state=optimizer_state))

class VersionedModelCheckpoint(Callback):
    def __init__(self, filepath, max_epochs=10000, **kwargs):
        kwargs['save_best_only'] = False
        self.delegate = keras.callbacks.ModelCheckpoint(filepath, **kwargs)
        self.filepath = filepath
        self.basepath, self.ext = os.path.splitext(filepath)
        self.epoch = 0
        width = int(np.log10(max_epochs)) + 1
        self.fmt_string = '{basepath}-{epoch:0' + str(width) + 'd}{ext}'

    def on_epoch_end(self, epoch, logs={}):
        logs['val_loss'] = -self.epoch
        self.delegate.on_epoch_end(epoch, logs)

        if os.path.exists(self.filepath):
            newpath = self.fmt_string.format(
                    basepath=self.basepath, epoch=self.epoch, ext=self.ext)
            os.rename(self.filepath, newpath)
        self.epoch += 1

    def _set_model(self, model):
        self.model = model
        self.delegate._set_model(model)

class SingleStepLearningRateSchedule(keras.callbacks.Callback):
    def __init__(self, patience=5, learning_rate_divisor=10.):
        self.patience = patience
        self.learning_rate_divisor = learning_rate_divisor
        self.best_loss = np.inf
        self.best_epoch = 0
        self.updated_lr = False

    def on_epoch_end(self, epoch, logs={}):
        if self.updated_lr:
            return

        if logs['val_loss'] < self.best_loss:
            self.best_loss = logs['val_loss']
            self.best_epoch = epoch

        if epoch - self.best_epoch > self.patience:
            old_lr = self.model.optimizer.lr.get_value()
            new_lr = (old_lr / self.learning_rate_divisor).astype(np.float32)
            print('old_lr', old_lr, 'new_lr', new_lr)
            self.model.optimizer.lr.set_value(new_lr)
            self.learning_rate_divisor = 1.
