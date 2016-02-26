# coding=utf-8

import urllib
import sys

def call_back(num, size, all):
    per = 100.0 * num * size / all
    if per > 100:
        per = 100
    sys.stdout.write('\r%f%% completed' % per)
    sys.stdout.flush()

def download_mnist():
    # download mnist dataset from lisa-lab website
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    local = '../data/mnist/mnist.pkl.gz'
    urllib.urlretrieve(url, local, call_back)
