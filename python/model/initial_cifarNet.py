import sys
sys.path.append("..")
from cnn.conv_net import ConvNet

def initial_cifar():
    # initial cifar net
    cnn = ConvNet()

    conv1_params = {'HF': 5, 'WF': 5, 'DF': 3, 'NF': 32, 'stride': 1, 'pad': 2, 'var': 0.01}
    cnn.add_layer('conv', conv1_params)
    pooling1_params = {'HF': 3, 'WF': 3, 'stride': 2, 'pad': [0, 1, 0, 1]}
    cnn.add_layer('max_pooling', pooling1_params)
    cnn.add_layer('relu', {})

    conv2_params = {'HF': 5, 'WF': 5, 'DF': 32, 'NF': 32, 'stride': 1, 'pad': 2, 'var': 0.02}
    cnn.add_layer('conv', conv2_params)
    cnn.add_layer('relu', {})
    pooling2_params = {'HF': 3, 'WF': 3, 'stride': 2, 'pad': [0, 1, 0, 1]}
    cnn.add_layer('max_pooling', pooling2_params)

    conv3_params = {'HF': 5, 'WF': 5, 'DF': 32, 'NF': 64, 'stride': 1, 'pad': 2, 'var': 0.03}
    cnn.add_layer('conv', conv3_params)
    cnn.add_layer('relu', {})
    pooling3_params = {'HF': 3, 'WF': 3, 'stride': 2, 'pad': [0, 1, 0, 1]}
    cnn.add_layer('max_pooling', pooling3_params)

    conv4_params = {'HF': 4, 'WF': 4, 'DF': 64, 'NF': 64, 'stride': 1, 'pad': 0, 'var': 0.04}
    cnn.add_layer('conv', conv4_params)
    cnn.add_layer('relu', {})

    conv5_params = {'HF': 1, 'WF': 1, 'DF': 64, 'NF': 10, 'stride': 1, 'pad': 0, 'var': 0.05}
    cnn.add_layer('conv', conv5_params)

    cnn.add_layer('softmax-loss', {})

    return cnn
