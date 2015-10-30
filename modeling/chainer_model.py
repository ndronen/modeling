import chainer
import chainer.functions as F
from chainer import optimizers

class Model(object):
    def __init__(self, args):
        for k,v in vars(args).iteritems():
            self.__dict__[k] = v
        self.init_params()
        self.init_optimizer()
        self.optimizer.setup(self.params)

    def init_optimizer(self):
        if self.optimizer == 'SGD':
            self.optimizer = optimizers.MomentumSGD(
                lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer == 'AdaDelta':
            self.optimizer = optimizers.AdaDelta()
        elif self.optimizer == 'AdaGrad':
            self.optimizer = optimizers.AdaGrad()
        elif self.optimizer == 'Adam':
            self.optimizer = optimizers.Adam()
        elif self.optimizer == 'RMSprop':
            self.optimizer = optimizers.RMSprop()

    def update(self):
        if hasattr(self, 'weight_decay'):
            if self.weight_decay > 0:
                self.optimizer.weight_decay(self.weight_decay)
        self.optimizer.update()

    def iteration(self, data, target, train=False):
        if train:
            self.optimizer.zero_grads()
        pred = self.forward(data)
        loss, metric = self.loss(pred, target)
        if train:
            loss.backward()
            self.update()
        return pred, loss, metric

    def fit(self, data, target):
        pred, loss, metric = self.iteration(data, target, train=True)
        return pred, loss, metric

    def evaluate(self, data, target):
        pred, loss, metric = self.iteration(data, target)
        return pred, loss, metric

    def init_params(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def loss(self, pred, target):
        raise NotImplementedError()

    def predict(self, data, target=None):
        raise NotImplementedError()

    def predict_proba(self, data):
        raise NotImplementedError()

    def to_gpu(self):
        self.params.to_gpu()

    def to_cpu(self):
        self.params.to_cpu()

class Classifier(Model):
    def loss(self, pred, target):
        target = chainer.Variable(target)
        loss = F.softmax_cross_entropy(pred, target)
        metric = F.accuracy(pred, target)
        return loss, metric

    def predict(self, data, target=None):
        pred = self.forward(data, train=False)
        if target is None:
            return np.argmax(F.softmax(pred).data, axis=1)
        else:
            loss, metric = self.loss(pred, target)
            return pred, loss, metric

    def predict_proba(self, data):
        pred = self.forward(data, train=False)
        return F.softmax(pred).data

class Regressor(Model):
    def loss(self, pred, target):
        target = chainer.Variable(target)
        loss = F.mean_squared_error(pred, target)
        return loss, loss

    def predict(self, data, target=None):
        pred = self.forward(data, train=False)
        if target is None:
            return pred
        else:
            loss, metric = self.loss(pred, target)
            return pred, loss, metric
