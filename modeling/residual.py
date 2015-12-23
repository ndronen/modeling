from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Layer, Dropout
from keras.activations import relu

class Identity(Layer):
    def get_output(self, train):
        return self.get_input(train)

def build_residual_block(name, input_shape, n_hidden, n_skip=2):
    """
    Rough sketch of building blocks of layers for residual learning.
    See http://arxiv.org/abs/1512.03385 for motivation.
    """
    block = Graph()
    input_name = 'x'
    block.add_input(input_name, input_shape=input_shape)

    # The current keras graph implementation doesn't allow you to connect
    # an input node to an output node.  Use Identity to work around that.
    block.add_node(Identity(), name=name+'identity', input=input_name)

    prev_output = input_name
    for i in range(n_skip):
        layer_name = 'h' + str(i)
        l = Dense(n_hidden, activation='relu')
        block.add_node(l, name=layer_name, input=prev_output)
        prev_output = layer_name
        if i < n_skip:
            block.add_node(Dropout(0.5), name=layer_name+'do', input=layer_name)
            prev_output = layer_name+'do'

    block.add_output(name=name+'output', inputs=[name+'identity', prev_output], merge_mode='sum')

    return block
