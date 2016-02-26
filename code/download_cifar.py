# coding=utf-8

import urllib
import sys
import tarfile

def call_back(num, size, all):
    per = 100.0 * num * size / all
    if per > 100:
        per = 100
    sys.stdout.write('\r%f%% completed' % per)
    sys.stdout.flush()

def untar(file_name, dirs):
    t = tarfile.open(file_name)
    t.extractall(path=dirs)

def download_cifar():
    # download cifar dataset
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    local = '../data/cifar/cifar-10-python.tar.gz'
    urllib.urlretrieve(url, local, call_back)
    untar(local, '../data/cifar')
