import os
import numpy as np
import keras
from keras.callbacks import Callback, EarlyStopping
import keras.callbacks
import numpy as np
import six
from sklearn.metrics import (classification_report, 
        confusion_matrix, f1_score, fbeta_score)

def predict(model, x, marshaller, batch_size=128):
    if isinstance(model, keras.models.Graph):
        if marshaller is None:
            raise ValueError("a marshaller is required with Graphs")
        x = marshaller.marshal(x)
        output = model.predict(x, batch_size=batch_size)
        y_hat = marshaller.unmarshal(output)
        y_hat = np.argmax(y_hat, axis=1)
    else:
        y_hat = model.predict_classes(x, verbose=0, batch_size=batch_size)
    return y_hat

class PredictionCallback(Callback):
    def __init__(self, x, logger, marshaller=None, iteration_freq=10, batch_size=128):
        self.__dict__.update(locals())
        self.callbacks = []

    def add(self, callback):
        self.callbacks.append(callback)

    def _set_model(self, model):
        self.model = model
        for cb in self.callbacks:
            cb._set_model(model)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        if 'iteration' in logs.keys() and logs['iteration'] % self.iteration_freq != 0:
            # If we've broken a large training set into smaller chunks, we don't
            # need to run the classification report after every chunk.
            return

        y_hat = predict(self.model, self.x, self.marshaller, batch_size=self.batch_size)
        logs['y_hat'] = y_hat
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, logs)

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

class DelegatingMetricCallback(Callback):
    def __init__(self, x, y, logger, metric_name, delegate, marshaller=None, batch_size=128):
        self.__dict__.update(locals())
        del self.self

    def _set_model(self, model):
        self.model = model
        self.delegate._set_model(model)

    def on_epoch_end(self, epoch, logs={}):
        try:
            y_hat = logs['y_hat']
        except KeyError:
            y_hat = predict(self.model, self.x, self.marshaller, batch_size=self.batch_size)
        metric = self.build_metric(logs)
        logs[self.metric_name] = metric(self.y, y_hat)
        self.logger('%s %.03f' % (self.metric_name, logs[self.metric_name]))
        self.delegate.on_epoch_end(epoch, logs)

    def build_metric(self, logs):
        return {
                'val_loss': lambda y,y_hat: logs['val_loss'],
                'val_acc': lambda y,y_hat: logs['val_acc'],
                'val_f1': f1_score,
                'val_f1': lambda y,y_hat: fbeta_score(y, y_hat, beta=0.5),
                'val_f2': lambda y,y_hat: fbeta_score(y, y_hat, beta=2)
                }[self.metric_name]

class ConfusionMatrix(Callback):
    def __init__(self, x, y, logger, marshaller=None, batch_size=128):
        self.__dict__.update(locals())
        del self.self

    def on_epoch_end(self, epoch, logs={}):
        try:
            y_hat = logs['y_hat']
        except KeyError:
            y_hat = predict(self.model, self.x, self.marshaller, batch_size=self.batch_size)
        self.logger('\nConfusion matrix')
        self.logger(confusion_matrix(self.y, y_hat))

class ClassificationReport(Callback):
    def __init__(self, x, y, logger, target_names=None, marshaller=None, batch_size=128):
        self.__dict__.update(locals())
        del self.self

        self.labels = np.arange(max(y)+1)

        if target_names is None:
            self.target_names = [str(t) for t in self.labels]
        else:
            self.target_names = [str(tn) for tn in target_names]

    def on_epoch_end(self, epoch, logs={}):
        try:
            y_hat = logs['y_hat']
        except KeyError:
            y_hat = predict(self.model, self.x, self.marshaller, batch_size=self.batch_size)

        self.logger('\nClassification report')
        self.logger(classification_report(
                self.y, y_hat,
                labels=self.labels, target_names=self.target_names))

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
