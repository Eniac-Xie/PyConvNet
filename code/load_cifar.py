# coding = utf-8

import os
import cPickle
import numpy as np
import scipy.io as sio
import download_cifar

from sklearn.cross_validation import train_test_split

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data():
    if os.path.exists('../data/cifar/cifar-10-python.tar.gz'):
        pass
    else:
        print 'download cifar dataset...'
        download_cifar.download_cifar()
    data = []
    labels = []
    for batch_idx in np.arange(5):
        data_path = '../data/cifar/cifar-10-batches-py/data_batch_%d' % (batch_idx + 1)
        data_batch = unpickle(data_path)
        sub_data = data_batch['data']
        sub_labels = np.array(data_batch['labels'])
        if batch_idx == 0:
            data = sub_data
            labels = sub_labels
        else:
            data = np.vstack((data, sub_data))
            labels = np.hstack((labels, sub_labels))
    labels = labels.reshape((-1, 1))
    return data, labels

def load_cifar():
    train_data, train_labels = load_data()
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42)
    train_data = train_data.reshape((-1, 32, 32, 3))
    train_data = np.transpose(train_data, (1, 2, 3, 0))
    train_labels = np.transpose(train_labels, (1, 0))
    valid_data = valid_data.reshape((-1, 32, 32, 3))
    valid_data = np.transpose(valid_data, (1, 2, 3, 0))
    valid_labels = np.transpose(valid_labels, (1, 0))

    # remove mean
    print 'remove mean'
    mean_img = np.mean(train_data, axis=3, keepdims=True)
    train_data = train_data - mean_img
    # mean_img = np.mean(valid_data, axis=3, keepdims=True)
    valid_data = valid_data - mean_img

    # normalize by image mean and std
    print 'normalize by image mean and std'
    train_data = train_data.reshape((3072, -1))
    valid_data = valid_data.reshape((3072, -1))

    train_data = train_data - np.mean(train_data, axis=0, keepdims=True)
    valid_data = valid_data - np.mean(valid_data, axis=0, keepdims=True)

    train_std = np.std(train_data, axis=0, keepdims=True)
    train_data = train_data * np.mean(train_std) / np.maximum(train_std, 40 * np.ones_like(train_std))
    train_data = train_data.reshape((32, 32, 3, -1))

    valid_std = np.std(valid_data, axis=0, keepdims=True)
    valid_data = valid_data * np.mean(valid_std) / np.maximum(valid_std, 40 * np.ones_like(valid_std))
    valid_data = valid_data.reshape((32, 32, 3, -1))

    # # zca whitening
    print 'zca whitening'
    train_data = train_data.reshape((3072, -1))
    valid_data = valid_data.reshape((3072, -1))
    train_cov = train_data.dot(train_data.T) / train_data.shape[1]
    D, V = np.linalg.eig(train_cov)
    # D = D.reshape((-1, 1))
    # d2 = np.diag(D)
    en = np.sqrt(np.mean(D))
    train_data = V.dot(np.diag(en / np.maximum(np.sqrt(D), 10 * np.ones_like(D)))).dot(V.T).dot(train_data)
    valid_data = V.dot(np.diag(en / np.maximum(np.sqrt(D), 10 * np.ones_like(D)))).dot(V.T).dot(valid_data)
    train_data = train_data.reshape((32, 32, 3, -1))
    valid_data = valid_data.reshape((32, 32, 3, -1))

    # save to file
    sio.savemat('../data/cifar/cifar.mat', {'train_data': train_data,
                                            'valid_data': valid_data,
                                            'train_labels': train_labels,
                                            'valid_labels': valid_labels})
    print 'preprocess done'
    return train_data, valid_data, train_labels, valid_labels
