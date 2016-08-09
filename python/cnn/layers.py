# coding=utf-8

import pylayer

import numpy as np
from im2col import im2col
from im2col import col2im

def conv_forward(x, w, b, params):
    # get convolution parameters
    stride = params['stride']
    pad = params['pad']
    # get input size
    H, W, D, N = x.shape
    HF, WF, DF, NF = w.shape
    _, _, DB, NB = b.shape
    # check input size
    assert D == DF, 'dimension does not work'
    assert NF == NB, 'batch size does not work'
    # check params
    assert (H + 2 * pad - HF) % stride == 0, 'pad and stride do not work'
    assert (W + 2 * pad - WF) % stride == 0, 'pad and stride do not work'
    # get output size
    HO = (H + 2 * pad - HF) / stride + 1
    WO = (W + 2 * pad - WF) / stride + 1
    x_col = im2col(x, HF, WF, pad, stride)
    w_col = w.transpose(3, 0, 1, 2).reshape((NF, -1))
    output_col = w_col.dot(x_col) + b.reshape(-1, 1)
    output_col = output_col.reshape((NF, HO, WO, N))
    output_col = output_col.transpose(1, 2, 0, 3)
    return output_col

# def conv_forward(x, w, b, params):
#     # get convolution parameters
#     stride = params['stride']
#     pad = params['pad']
#     # get input size
#     H, W, D, N = x.shape
#     HF, WF, DF, NF = w.shape
#
#     _, _, DB, NB = b.shape
#
#     stride_h, stride_w = stride, stride
#     pad_h, pad_w = pad, pad
#     kernel_h, kernel_w = HF, WF
#     # check input size
#     assert D == DF, 'dimension does not work'
#     assert NF == NB, 'batch size does not work'
#     # check params
#     assert (H + 2 * pad - HF) % stride == 0, 'pad and stride do not work'
#     assert (W + 2 * pad - WF) % stride == 0, 'pad and stride do not work'
#     # get output size
#     HO = (H + 2 * pad - HF) / stride + 1
#     WO = (W + 2 * pad - WF) / stride + 1
#
#     # initial data
#     input_data = x.transpose(3, 2, 0, 1).astype(np.float32)
#     input_data = input_data.copy(order='C')
#     input_tensor = pylayer.PyTensor(N, D, H, W)
#     input_tensor.init_from_numpy(input_data)
#
#     filter_data = w.transpose(3, 2, 0, 1).astype(np.float32)
#     filter_data = filter_data.copy(order='C')
#     filter_tensor = pylayer.PyTensor(NF, DF, HF, WF)
#     filter_tensor.init_from_numpy(filter_data)
#
#     bias_data = b.transpose(3, 2, 0, 1).astype(np.float32)
#     bias_data = bias_data.copy(order='C')
#     bias_tensor = pylayer.PyTensor(NB, 1, 1, 1)
#     bias_tensor.init_from_numpy(bias_data)
#
#     output_data = np.zeros([N, NF, HO, WO], dtype=np.float32)
#     output_data = output_data.copy(order='C')
#     output_tensor = pylayer.PyTensor(N, NF, HO, WO)
#     output_tensor.init_from_numpy(output_data)
#
#     input_vec = pylayer.PyTensorVec()
#     input_vec[:] = [input_tensor, filter_tensor, bias_tensor]
#
#     output_vec = pylayer.PyTensorVec()
#     output_vec[:] = [output_tensor]
#
#     L1 = pylayer.PyConvolutionLayer(pad_h, pad_w, kernel_h, kernel_w, stride_h, stride_w)
#     L1.forward(input_vec, output_vec)
#
#     output_vec[0].return_numpy(output_data)
#     output_data = output_data.transpose(2, 3, 1, 0)
#     return output_data


def conv_backward(x, w, b, conv_param, dout):
    HF, WF, DF, NF = w.shape
    x_col = im2col(x, HF, WF, conv_param['pad'], conv_param['stride'])
    w_col = w.transpose(3, 0, 1, 2).reshape((NF, -1))
    db = np.sum(dout, axis=(0, 1, 3))
    dout = dout.transpose(2, 0, 1, 3)
    dout = dout.reshape((w_col.shape[0], x_col.shape[-1]))
    dx_col = w_col.T.dot(dout)
    dw_col = dout.dot(x_col.T)

    dx = col2im(dx_col, x.shape, HF, WF, conv_param['pad'], conv_param['stride'])
    dw = dw_col.reshape((dw_col.shape[0], HF, WF, DF))
    dw = dw.transpose(1, 2, 3, 0)

    return [dx, dw, db]

def max_pooling_forward(x, pool_params):
    # get max-pooling parameters
    stride = pool_params['stride']
    HF = pool_params['HF']
    WF = pool_params['WF']
    pad = pool_params['pad']
    # get input size
    H, W, D, N = x.shape
    x_reshaped = x.reshape(H, W, 1, -1)
    # get output size
    HO = 0
    WO = 0
    if type(pad) is int:
        HO = (H + 2 * pad - HF) / stride + 1
        WO = (W + 2 * pad - WF) / stride + 1
    else:
        HO = (H + pad[0] + pad[1] - HF) / stride + 1
        WO = (W + pad[2] + pad[3] - WF) / stride + 1
    x_col = im2col(x_reshaped, HF, WF, pad, stride)
    x_col_argmax = np.argmax(x_col, axis=0)
    x_col_max = x_col[x_col_argmax, np.arange(x_col.shape[1])]
    out = x_col_max.reshape((HO, WO, D, N))
    return out

def max_pooling_backward(x, dout, pool_params):
    H, W, D, N = x.shape
    x_reshaped = x.reshape(H, W, 1, -1)
    x_col = im2col(x_reshaped, pool_params['HF'],
                   pool_params['WF'], pool_params['pad'], pool_params['stride'])
    x_col_argmax = np.argmax(x_col, axis=0)
    dx_col = np.zeros_like(x_col)
    dx_col[x_col_argmax, np.arange(x_col.shape[1])] = dout.ravel()
    dx_shaped = col2im(dx_col, x_reshaped.shape, pool_params['HF'], pool_params['WF'],
                       pool_params['pad'], stride=pool_params['stride'])
    dx = dx_shaped.reshape(x.shape)
    return [dx]

def relu_forward(x):
    out = np.where(x > 0, x, 0)
    return out

def relu_backward(x, dout):
    dx = np.where(x > 0, dout, 0)
    return [dx]

def softmax_loss_forward(x, y):
    # x is the prediction(C * N), y is the label(1 * N)
    x_reshaped = x.reshape((x.shape[2], x.shape[3]))
    probs = np.exp(x_reshaped - np.max(x_reshaped, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    N = x_reshaped.shape[1]
    loss = -np.sum(np.log(probs[y, np.arange(N)])) / N
    return loss

def softmax_loss_backward(x, y):
    x_reshaped = x.reshape((x.shape[2], x.shape[3]))
    probs = np.exp(x_reshaped - np.max(x_reshaped, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    dx = probs.copy()
    N = x_reshaped.shape[1]
    dx[y, np.arange(N)] -= 1
    dx /= N
    dx = dx.reshape((1, 1, dx.shape[0], dx.shape[1]))
    return [dx]

def softmax(x):
    x_reshaped = x.reshape((x.shape[2], x.shape[3]))
    probs = np.exp(x_reshaped - np.max(x_reshaped, axis=0, keepdims=True))
    probs /= np.sum(probs, axis=0, keepdims=True)
    return np.argmax(probs, axis=0)
