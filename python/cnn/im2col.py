# coding=utf-8

import numpy as np

def im2col_index(x_shape, HF, WF, pad, stride):
    # get input size
    H, W, D, N = x_shape
    # get output size
    out_h = 0
    out_w = 0
    if type(pad) is int:
        out_h = (H + 2 * pad - HF) / stride + 1
        out_w = (W + 2 * pad - WF) / stride + 1
    else:
        out_h = (H + pad[0] + pad[1] - HF) / stride + 1
        out_w = (W + pad[2] + pad[3] - WF) / stride + 1
    # for row index, compute the first index of the first HF * WF block
    r0 = np.repeat(np.arange(HF), WF)
    r0 = np.tile(r0, D)
    # then compute the bias of each block
    r_bias = stride * np.repeat(np.arange(out_h), out_w)
    # then the row index is the r0 + r_bias
    r = r0.reshape(-1, 1) + r_bias.reshape(1, -1)

    # the same to the col index
    c0 = np.tile(np.arange(WF), HF * D)
    c_bias = stride * np.tile(np.arange(out_w), out_h)
    c = c0.reshape(-1, 1) + c_bias.reshape(1, -1)

    # then the dimension index
    d = np.repeat(np.arange(D), HF * WF).reshape(-1, 1)

    return (r, c, d)

def im2col(x, HF, WF, pad, stride):
    # padding
    x_padded = None
    if type(pad) is int:
        x_padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    else:
        x_padded = np.pad(x, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0), (0, 0)), mode='constant')
    r, c, d = im2col_index(x.shape, HF, WF, pad, stride)
    cols = x_padded[r, c, d, :]
    cols = cols.reshape(HF * WF * x.shape[2], -1)
    return cols

def col2im(cols, x_shape, HF, WF, pad, stride):
    # get input size
    H, W, D, N = x_shape
    H_padded = 0
    W_padded = 0
    if type(pad) is int:
        H_padded, W_padded = H + 2 * pad, W + 2 * pad
    else:
        H_padded, W_padded = H + pad[0] + pad[1], W + pad[2] + pad[3]
    x_padded = np.zeros((H_padded, W_padded, D, N), dtype=cols.dtype)
    r, c, d = im2col_index(x_shape, HF, WF, pad, stride)
    cols_reshaped = cols.reshape((HF * WF * D, -1, N))
    np.add.at(x_padded, (r, c, d, slice(None)), cols_reshaped)
    if pad == 0:
        return x_padded
    elif type(pad) is int:
        return x_padded[pad:-pad, pad:-pad, :, :]
    else:
        return x_padded[pad[0]:-pad[1], pad[2]:-pad[3], :, :]
