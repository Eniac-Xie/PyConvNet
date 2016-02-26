# coding=utf-8

# reading mnist dataset
import cPickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import sys
sys.path.append("..")
from gaussian_kernel import gkern
from cnn.layers import conv_forward
from cnn.layers import conv_backward
from cnn.layers import max_pooling_forward
from cnn.layers import max_pooling_backward
from cnn.layers import relu_forward
from cnn.layers import relu_backward
from cnn.layers import softmax_loss_forward
from cnn.layers import softmax_loss_backward

from util import numerical_gradient
from util import numerical_gradient_loss
from util import rel_error
# Load the dataset
f = gzip.open('../../data/mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
# train_set, valid_set, test_set are all tuple with two elements: data and label

# select one image and test the conv_forward layer by gaussian filtering
im = train_set[0][0, :]
im = im.reshape((28, 28)) * 255
plt.subplot(1, 2, 1)
plt.imshow(im.astype('uint8'), cmap=cm.Greys_r)

g_k = gkern(4, 1)
im = im.reshape(28, 28, 1, 1)
filter = g_k.reshape(4, 4, 1, 1)
b = np.zeros((1, 1, 1, 1))
params = {'stride': 2, 'pad': 2}
res = conv_forward(im, filter, b, params)
res = res.reshape(res.shape[0], res.shape[1])
plt.subplot(1, 2, 2)
plt.imshow(res.astype('uint8'), cmap=cm.Greys_r)
plt.show()

# test conv_backward
x = np.random.randn(5, 5, 3, 4)
w = np.random.randn(3, 3, 3, 2)
b = np.random.randn(1, 1, 1, 2)
dout = np.random.randn(5, 5, 2, 4)
conv_param = {'stride': 1, 'pad': 1}

dx_num = numerical_gradient(lambda x: conv_forward(x, w, b, conv_param), x, dout)
dw_num = numerical_gradient(lambda w: conv_forward(x, w, b, conv_param), w, dout)
db_num = numerical_gradient(lambda b: conv_forward(x, w, b, conv_param), b, dout)

out = conv_forward(x, w, b, conv_param)
dx, dw, db = conv_backward(x, w, b, conv_param, dout)

# Your errors should be around 1e-9'
print 'Testing conv_backward function'
print 'dx error: ', rel_error(dx, dx_num)
print 'dw error: ', rel_error(dw, dw_num)
print 'db error: ', rel_error(db, db_num)

x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'HF': 2, 'WF': 2, 'stride': 2}
x = x.transpose(2, 3, 1, 0)
out = max_pooling_forward(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])
correct_out = correct_out.transpose(2, 3, 1, 0)
# Compare your output with ours. Difference should be around 1e-8.
print 'Testing max_pooling_forward function:'
print 'difference: ', rel_error(out, correct_out)

x = np.random.randn(8, 8, 3, 2)
dout = np.random.randn(4, 4, 3, 2)
pool_param = {'HF': 2, 'WF': 2, 'stride': 2}

dx_num = numerical_gradient(lambda x: max_pooling_forward(x, pool_param), x, dout)

out = max_pooling_forward(x, pool_param)
dx = max_pooling_backward(x, dout, pool_param)

# Your error should be around 1e-12
print 'Testing max_pool_backward_naive function:'
print 'dx error: ', rel_error(dx, dx_num)

x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

out = relu_forward(x)
correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                        [ 0.,          0.,          0.04545455,  0.13636364,],
                        [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

# Compare your output with ours. The error should be around 1e-8
print 'Testing relu_forward function:'
print 'difference: ', rel_error(out, correct_out)

x = np.random.randn(10, 10)
dout = np.random.randn(*x.shape)

dx_num = numerical_gradient(lambda x: relu_forward(x), x, dout)

dx = relu_backward(x, dout)

# The error should be around 1e-12
print 'Testing relu_backward function:'
print 'dx error: ', rel_error(dx_num, dx)

# num_classes, num_inputs = 10, 50
# x = 0.001 * np.random.randn(num_classes, num_inputs)
# y = np.random.randint(num_classes, size=num_inputs)
#
# dx_num = numerical_gradient_loss(lambda x: softmax_loss_forward(x, y), x, verbose=False)
# loss = softmax_loss_forward(x, y)
# dx = softmax_loss_backward(x, y)
#
# # Test softmax_loss function. Loss should be 2.3 and dx error should be 1e-8
# print '\nTesting softmax_loss:'
# print 'loss: ', loss
# print 'dx error: ', rel_error(dx_num, dx)
