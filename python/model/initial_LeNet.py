import sys
sys.path.append("..")
from cnn.conv_net import ConvNet

def initial_LeNet():
    # initial LeNet
    cnn = ConvNet()

    conv1_params = {'HF': 5, 'WF': 5, 'DF': 1, 'NF': 20, 'stride': 1, 'pad': 0, 'var': 0.01}
    cnn.add_layer('conv', conv1_params)

    pooling1_params = {'HF': 2, 'WF': 2, 'stride': 2, 'pad': 0}
    cnn.add_layer('max_pooling', pooling1_params)

    conv2_params = {'HF': 5, 'WF': 5, 'DF': 20, 'NF': 50, 'stride': 1, 'pad': 0, 'var': 0.01}
    cnn.add_layer('conv', conv2_params)

    pooling2_params = {'HF': 2, 'WF': 2, 'stride': 2, 'pad': 0}
    cnn.add_layer('max_pooling', pooling2_params)

    conv3_params = {'HF': 4, 'WF': 4, 'DF': 50, 'NF': 500, 'stride': 1, 'pad': 0, 'var': 0.01}
    cnn.add_layer('conv', conv3_params)

    cnn.add_layer('relu', {})

    conv4_params = {'HF': 1, 'WF': 1, 'DF': 500, 'NF': 10, 'stride': 1, 'pad': 0, 'var': 0.01}
    cnn.add_layer('conv', conv4_params)

    cnn.add_layer('softmax-loss', {})

    return cnn
