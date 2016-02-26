# coding = utf-8

import numpy as np

# this two function are from stanford cs231n: http://cs231n.stanford.edu/syllabus.html
# they are used to check gradient computing and get relative error

def numerical_gradient(f, x, df, h=1e-5):
    # initialize grad
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x)
        x[ix] = oldval - h
        neg = f(x)
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def rel_error(x, y):
    # returns relative error
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def numerical_gradient_loss(f, x, verbose=True, h=0.00001):
    # a naive implementation of numerical gradient of f at x
    # - f should be a function that takes a single argument
    # - x is the point (numpy array) to evaluate the gradient at

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
          print ix, grad[ix]
        it.iternext() # step to next dimension

    return grad