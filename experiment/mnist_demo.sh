cd ../data/mnist/

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip -c train-images-idx3-ubyte.gz > train-images-idx3-ubyte
gunzip -c t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte
gunzip -c train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte
gunzip -c t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte

cd ../../build/bin
./MnistDemo
