# PyConvNet: CNN for Python
**PyConvNet** is a python implementation of convolutional neural network.

To train LeNet on MNIST dataset, just do as follow(you may need some python package such as numpy matplotlib):

1. cd python
2. python mnist_demo.py

A C++ version ConNet is also available now. It is more faster than the Python version. To try the C++ implemention, you will need Openblas and boost installed, and then do as follow:

1. sh build.sh
2. cd experiment/
3. run mnist_demo.sh

Then the script will download the mnist dataset and train the lenet.

This is a brief CNN tutorial (from [Jianxin Wu](http://cs.nju.edu.cn/wujx/)'s homepage):

http://cs.nju.edu.cn/_upload/tpl/00/ed/237/template237/paper/CNN.pdf
