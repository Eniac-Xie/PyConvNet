# include <pooling_layer.hpp>
# include <cfloat>
# include <vector>
# include <cstring>

void PoolingLayer::forward(std::vector<Tensor> &input, std::vector<Tensor> &output) {
    Tensor input_data(input[0]);
    Tensor output_data(output[0]);

    int N_in = input_data.get_N();
    int C_in = input_data.get_C();
    int H_in = input_data.get_H();
    int W_in = input_data.get_W();

    int N_out = output_data.get_N();
    int C_out = output_data.get_C();
    int H_out = output_data.get_H();
    int W_out = output_data.get_W();

    float* input_data_ptr = input_data.get_data().get();
    float* output_data_ptr = output_data.get_data().get();
    float* output_data_index = new float[N_out * C_out * H_out * W_out];

    for (int n = 0; n < N_in; ++n) {
        for(int c = 0; c < C_in; ++c) {
            for(int h = -pad_h; h < H_in + pad_h - kernel_h; h += stride_h) {
                for(int w = -pad_w; w < W_in + pad_w - kernel_w; w += stride_w) {
                    float current_max = -FLT_MAX;
                    int max_index = 0;
                    int base_pos = n * C_in * H_in * W_in +
                                   c * H_in * W_in + h * W_in + w;
                    for(int bias_h = 0; bias_h < kernel_h; ++bias_h) {
                        for(int bias_w = 0; bias_w < kernel_w; ++bias_w) {
                            if(h + bias_h < 0 || w + bias_w < 0 || h + bias_h > H_in ||
                                    w + bias_w > W_in) {
                                continue;
                            }
                            int bias_pos = bias_h * W_in + bias_w;
                            if(input_data_ptr[base_pos + bias_pos] > current_max) {
                                max_index = bias_h * W_in + bias_w;
                                current_max = input_data_ptr[base_pos + bias_pos];
                            }
                        }
                    }
                    int dest_pos = n * C_out * H_out * W_out +
                            c * H_out * W_out + (h + pad_h) / stride_h * W_out +
                            (w + pad_w) / stride_w;
                    output_data_ptr[dest_pos] = current_max;
                    output_data_index[dest_pos] = max_index;
                }
            }
        }
    }
    max_index_mask.assign(output_data_index, output_data_index + N_out * C_out * H_out * W_out);
    delete[] output_data_index;
}

void PoolingLayer::backward(std::vector<Tensor> &input, std::vector<Tensor> &d_input,
                            std::vector<Tensor> &d_output) {
    Tensor d_input_data(d_input[0]);
    Tensor d_output_data(d_output[0]);

    int N_in = d_input_data.get_N();
    int C_in = d_input_data.get_C();
    int H_in = d_input_data.get_H();
    int W_in = d_input_data.get_W();

    int N_out = d_output_data.get_N();
    int C_out = d_output_data.get_C();
    int H_out = d_output_data.get_H();
    int W_out = d_output_data.get_W();

    float* d_input_data_ptr = d_input_data.get_data().get();
    memset(d_input_data_ptr, 0, N_in * C_in * H_in * W_in * sizeof(float));
    float* d_output_data_ptr = d_output_data.get_data().get();

    for (int n = 0; n < N_out; ++n) {
        for(int c = 0; c < C_out; ++c) {
            for(int h = 0; h < H_out; ++h) {
                for(int w = 0; w < W_out; ++w) {
                    int original_base_pos = n * C_in * H_in * W_in +
                        c * H_in * W_in + (h * stride_h - pad_h) * H_in +
                            w * stride_w - pad_w;
                    int original_bias_pos = max_index_mask[n * C_out * H_out * W_out +
                        c * H_out * W_out + h * W_out + w];
                    d_input_data_ptr[original_base_pos + original_bias_pos] +=
                    d_output_data_ptr[n * C_out * H_out * W_out +
                                      c * H_out * W_out + h * W_out + w];
                }
            }
        }
    }
}

