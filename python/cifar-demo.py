# coding = utf-8

import os
import scipy.io as sio
import numpy as np

from load_cifar import load_cifar
from model.initial_cifarNet import initial_cifar

train_data = []
valid_data = []
train_labels = []
valid_labels = []
if os.path.isfile('../data/cifar/cifar.mat'):
    print 'read mat file: %s' % ('../data/cifar/cifar.mat')
    data = sio.loadmat('../data/cifar/cifar.mat')
    train_data = data['train_data']
    valid_data = data['valid_data']
    train_labels = data['train_labels']
    valid_labels = data['valid_labels']
else:
    train_data, valid_data, train_labels, valid_labels = load_cifar()

cnn = initial_cifar()
lr = np.logspace(-2, -3, 20)
cnn.train(train_data, train_labels, lr, epoch=20, batch_size=32)

res = cnn.predict(valid_data)
res = res.reshape(valid_labels.shape)

print 'Accuracy is: %f' % (np.sum(res == valid_labels) / float(np.max(valid_labels.shape)))
