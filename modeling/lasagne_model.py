import lasagne
import theano.tensor as T

class Model(object):
    def build_input_var(self):
        # TODO: make this abstract.
        return T.dmatrix('input')

    def build_target_var(self):
        # TODO: make this abstract.
        return T.ivector('target')

    def __init__(self, args):
        for k,v in vars(args).iteritems():
            self.__dict__[k] = v

        self.input_var = self.build_input_var()
        self.target_var = self.build_target_var()

        self.model = self.build_model(self.input_var)
        self.output = lasagne.layers.get_output(self.model)
        # TODO: implement build_loss.
        self.loss = lasagne.objectives.categorical_crossentropy(
                self.output, self.target_var)

        self.params = lasagne.layers.get_all_params(network, trainable=True)
        # TODO: build updates.
        self.updates = lasagne.updates.nesterov_momentum(
            self.loss, self.params, learning_rate=0.01, momentum=0.9)

        # Theano variables for operations on held-out data.
        self.predict_var = lasagne.layers.get_output(self.model,
                deterministic=True)
        # TODO: build predict loss.
        self.predict_loss = lasagne.objectives.categorical_crossentropy(
                self.predict_var, self.target_var)
        self.predict_loss = self.predict_loss.mean()
        self.predict_accuracy = T.eq(
                T.argmax(self.predict_var, axis=1), self.target_var)
        self.predict_accuracy = T.mean(
                self.predict_accuracy_var, dtype=theano.config.floatX)

        self.train_fn = theano.function(
                [self.input_var, self.target_var],
                self.loss,
                updates=self.updates)

        self.val_fn = theano.function(
                [self.input_var, self.target_var],
                [self.predict_loss, self.predict_accuracy])
