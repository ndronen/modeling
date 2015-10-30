import numpy as np
import theano.tensor as tt
from theano import function

eps = 1e-20

def support(y):
    return y.sum(axis=0)

def true_positive(y, y_hat):
    return (tt.eq(y_hat, y) & tt.eq(y, 1)).sum(axis=0)

def make_y_diff(y, y_hat):
    return y_hat - y

def false_positive(y_diff):
    return tt.eq(y_diff, 1).sum(axis=0)

def true_negative(y_diff):
    return tt.eq(y_diff, 0).sum(axis=0)

def false_negative(y_diff):
    return tt.eq(y_diff, -1).sum(axis=0)

def precision(y, y_hat, eps=1e-9, y_diff=None):
    tp = true_positive(y, y_hat)
    if y_diff is None:
        y_diff = make_y_diff(y, y_hat)
    fp = false_positive(y_diff)
    return tp/(tp+fp+eps)

def recall(y, y_hat, eps=1e-9, y_diff=None):
    tp = true_positive(y, y_hat)
    if y_diff is None:
        y_diff = make_y_diff(y, y_hat)
    fn = false_negative(y_diff)
    return tp/(tp+fn+eps)

def fbeta_loss(y, y_hat, beta=0.5, eps=1e-9, average=None):
    """
    Returns the negative of the F_beta measure, because the
    optimizer is trying to minimize the objective.
    """
    y_diff = make_y_diff(y, y_hat)
    pr = precision(y, y_hat, eps=eps, y_diff=y_diff)
    rc = recall(y, y_hat, eps=eps, y_diff=y_diff)

    f_per_class = ( (1 + beta**2) * (pr * rc) ) / (beta**2 * pr + rc + eps)

    if average is None:
        f = f_per_class
    elif average == 'macro':
        f = f_per_class.mean()
    elif average == 'weighted':
        s = support(y)
        f = ((f_per_class * s) / s.sum()).sum()

    return -f


y = tt.matrix('y', dtype='int64')
y_hat = tt.matrix('y', dtype='int64')

floss = fbeta_loss(y, y_hat, average='weighted')
f = function([y, y_hat], floss)

loss = f(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]),
    np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))

print("loss", loss)
print("grad", tt.grad(loss, floss))

import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

###########################################################################
# Declare Theano symbolic variables
###########################################################################

x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

###########################################################################
# Construct Theano expression graph
###########################################################################

# Probability that target = 1
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   

# The prediction thresholded
prediction = p_1 > 0.5                    

# Cross-entropy loss function
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) 

# The cost to minimize
cost = xent.mean() + 0.01 * (w ** 2).sum()

# Compute the gradient of the cost (we shall return to this in a following
# section of this tutorial).
gw, gb = T.grad(cost, [w, b])             

# Compile
train = theano.function(
    inputs=[x,y],
    outputs=[prediction, xent],
    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
