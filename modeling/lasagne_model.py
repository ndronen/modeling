import lasagne
import theano.tensor as T
import theano

class Model(object):
    def build_input_var(self):
        raise NotImplementedError()

    def build_target_var(self):
        raise NotImplementedError()

    def build_updates(self):
        raise NotImplementedError()

    def __init__(self, args):
        for k,v in vars(args).iteritems():
            self.__dict__[k] = v

        self.input_var = self.build_input_var()
        self.target_var = self.build_target_var()

        self.model = self.build_model(self.input_var)

        self.train_output = lasagne.layers.get_output(self.model)
        self.train_loss = self.build_loss(self.train_output)
        self.params = lasagne.layers.get_all_params(self.model, trainable=True)
        self.updates = self.build_updates()

        self.test_output = lasagne.layers.get_output(self.model,
                deterministic=True)
        self.test_loss = self.build_loss(self.test_output)
        self.test_accuracy = T.eq(
                T.argmax(self.test_output, axis=1), self.target_var)
        self.test_accuracy = T.mean(
                self.test_accuracy, dtype=theano.config.floatX)

        self.train_fn = theano.function(
                [self.input_var, self.target_var],
                self.train_loss,
                updates=self.updates)

        self.val_fn = theano.function(
                [self.input_var, self.target_var],
                [self.test_loss, self.test_accuracy])

        self.pred_fun = theano.function([self.input_var], self.test_output)

    def fit(self, data, target):
        return self.train_fn(data, target)

    def evaluate(self, data, target):
        output = self.val_fn(data, target)
        return output[0], output[1]

    def predict(self, data):
        pred = self.pred_fn(data)
        return pred

class Classifier(Model):
    def build_loss(self, output):
        loss = lasagne.objectives.categorical_crossentropy(
                output, self.target_var)
        return loss.mean()
