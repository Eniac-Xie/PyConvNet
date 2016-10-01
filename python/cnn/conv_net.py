# coding = utf-8

from layers import *
import time

class ConvNet:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer_type, layer_params):
        if layer_type == 'conv':
            HF = layer_params['HF']
            WF = layer_params['WF']
            DF = layer_params['DF']
            NF = layer_params['NF']
            l_weights = np.random.normal(0, layer_params['var'], (HF, WF, DF, NF))
            l_bias = np.zeros((1, 1, 1, NF))
            layer_params = {'type': 'conv',
                            'weights': l_weights,
                            'bias': l_bias,
                            'stride': layer_params['stride'],
                            'pad': layer_params['pad'],
                            'input': None,
                            'output': None,
                            'grad': None}
            self.layers.append(layer_params)
        elif layer_type == 'max_pooling':
            layer_params = {'type': 'max_pooling',
                            'stride': layer_params['stride'],
                            'HF': layer_params['HF'],
                            'WF': layer_params['WF'],
                            'pad': layer_params['pad'],
                            'input': None,
                            'output': None,
                            'grad': None}
            self.layers.append(layer_params)
        elif layer_type == 'relu':
            layer_params = {'type': 'relu',
                            'input': None,
                            'output': None,
                            'grad': None}
            self.layers.append(layer_params)
        elif layer_type == 'softmax-loss':
            layer_params = {'type': 'softmax-loss',
                            'input': None,
                            'output': None,
                            'grad': None}
            self.layers.append(layer_params)
        else:
            print 'unkonw layer type!\n'
            exit(1)

    def forward(self, data, label=[]):
        for idx, each_layer in enumerate(self.layers):
            if idx == 0:
                each_layer['input'] = data
            else:
                each_layer['input'] = self.layers[idx - 1]['output']
            if each_layer['type'] == 'conv':
                params = {'stride': each_layer['stride'],
                          'pad': each_layer['pad']}
                params['pad'] = each_layer['pad']
                each_layer['output'] = conv_forward(each_layer['input'],
                                                    each_layer['weights'],
                                                    each_layer['bias'], params)
            elif each_layer['type'] == 'max_pooling':
                params = {'stride': each_layer['stride'],
                          'HF': each_layer['HF'],
                          'WF': each_layer['WF'],
                          'pad': each_layer['pad']
                          }
                each_layer['output'] = max_pooling_forward(each_layer['input'],
                                                           params)
            elif each_layer['type'] == 'relu':
                each_layer['output'] = relu_forward(each_layer['input'])
            elif each_layer['type'] == 'softmax-loss':
                if len(label) == 0:
                    each_layer['output'] = softmax(each_layer['input'])
                else:
                    each_layer['output'] = softmax_loss_forward(each_layer['input'], label)
                    return each_layer['output']

    def backward(self, data, label, lr=0.01):
        for idx in reversed(np.arange(len(self.layers))):
            current_layer = self.layers[idx]
            if current_layer['type'] == 'softmax-loss':
                if idx != len(self.layers) - 1:
                    print 'wrong architecture'
                    exit(-1)
                self.layers[idx]['grad'] = softmax_loss_backward(self.layers[idx - 1]['output'],
                                                                 label)

            elif current_layer['type'] == 'conv':
                conv_param = {'stride': self.layers[idx]['stride'],
                              'pad': self.layers[idx]['pad']}
                self.layers[idx]['grad'] = conv_backward(self.layers[idx]['input'],
                                                    self.layers[idx]['weights'],
                                                    self.layers[idx]['bias'],
                                                    conv_param,
                                                    self.layers[idx + 1]['grad'][0])
                self.layers[idx]['weights'] = self.layers[idx]['weights'] \
                                              - lr * self.layers[idx]['grad'][1]
                # the learning rate of bias is 2 times of weights'
                self.layers[idx]['bias'] = self.layers[idx]['bias'] \
                                              - 2 * lr * self.layers[idx]['grad'][2]
            elif current_layer['type'] == 'max_pooling':
                pooling_params = {'stride': self.layers[idx]['stride'],
                                  'HF': self.layers[idx]['HF'],
                                  'WF': self.layers[idx]['WF'],
                                  'pad': self.layers[idx]['pad']}
                self.layers[idx]['grad'] = max_pooling_backward(self.layers[idx]['input'],
                                                                self.layers[idx + 1]['grad'][0],
                                                                pooling_params)
            elif current_layer['type'] == 'relu':
                self.layers[idx]['grad'] = relu_backward(self.layers[idx]['input'], self.layers[idx + 1]['grad'][0])

    def predict(self, test_data, batch_size=50):
        _, _, _, N = test_data.shape
        prediction = np.zeros((1, N))
        batch_num = np.ceil(float(N) / float(batch_size))
        for batch_idx in np.arange(batch_num):
            sub_test_data = None
            sub_test_label= None
            if batch_idx == batch_num - 1:
                sub_test_data = test_data[:, :, :, batch_idx * batch_size:]
                self.forward(sub_test_data)
                prediction[0, batch_idx * batch_size:] = self.layers[-1]['output']
            else:
                sub_test_data = test_data[:, :, :, batch_idx * batch_size:(batch_idx + 1) * batch_size]
                self.forward(sub_test_data)
                prediction[0, batch_idx * batch_size:(batch_idx + 1) * batch_size] = self.layers[-1]['output']
        return prediction

    def train(self, train_data, train_label, lr, epoch=20, batch_size=100):
        H, W, D, N = train_data.shape
        _, N_l = train_label.shape
        assert N == N_l, 'Wrong data input!'
        # shuffle train_data
        shuffle_idx = np.arange(N)
        train_data = train_data[:, :, :, shuffle_idx]
        train_label = train_label[:, shuffle_idx]
        error_list = []
        for epoch_idx in np.arange(epoch):
            batch_num = np.ceil(float(N) / float(batch_size))
            for batch_idx in np.arange(batch_num):
                # start timing
                start_t = time.clock()
                sub_train_data = None
                sub_train_label= None
                if batch_idx == batch_num - 1:
                    sub_train_data = train_data[:, :, :, batch_idx * batch_size:]
                    sub_train_label = train_label[:, batch_idx * batch_size:]
                else:
                    sub_train_data = train_data[:, :, :, batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    sub_train_label = train_label[:, batch_idx * batch_size:(batch_idx + 1) * batch_size]

                loss = self.forward(sub_train_data, sub_train_label)
                error_list.append(loss)

                self.backward(sub_train_data, sub_train_label, lr[epoch_idx])
                # end timing
                end_t = time.clock()
                print 'epoch: %d, batch: %d, time: %f sec, obj: %f' % (epoch_idx,
                                                                       batch_idx,
                                                                       end_t - start_t,
                                                                       np.mean(np.array(error_list)))
