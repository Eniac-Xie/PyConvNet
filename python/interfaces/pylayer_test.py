import numpy as np
import pylayer

input_data = np.ones([200, 3, 50, 50], dtype=np.float32)
input_tensor = pylayer.PyTensor(200, 3, 50, 50)
input_tensor.init_from_numpy(input_data)

filter_data = np.ones([2, 3, 3, 3], dtype=np.float32)
filter_tensor = pylayer.PyTensor(2, 3, 3, 3)
filter_tensor.init_from_numpy(filter_data)

bias_data = np.ones([2, 1, 1, 1], dtype=np.float32)
bias_tensor = pylayer.PyTensor(2, 1, 1, 1)
bias_tensor.init_from_numpy(bias_data)

output_data = np.ones([200, 2, 25, 25], dtype=np.float32)
output_tensor = pylayer.PyTensor(200, 2, 25, 25)
output_tensor.init_from_numpy(output_data)

input_vec = pylayer.PyTensorVec()
input_vec[:] = [input_tensor, filter_tensor, bias_tensor]

output_vec = pylayer.PyTensorVec()
output_vec[:] = [output_tensor]

L1 = pylayer.PyConvolutionLayer(1, 1, 3, 3, 2, 2)
L1.forward(input_vec, output_vec)
output_vec[0].return_numpy(output_data)

print output_data
