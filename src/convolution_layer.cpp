#include <vector>
#include <algorithm>
#include <assert.h>
#include "convolution_layer.hpp"
#include "Tensor.hpp"
#include "im2col.hpp"
#include "blas_function.hpp"

void ConvolutionLayer::forward(std::vector<Tensor>& input,
                               std::vector<Tensor>& output ) {
    Tensor input_data(input[0]);
    Tensor filter(input[1]);
    Tensor bias(input[2]);
    Tensor output_data(output[0]);

    int N_in = input_data.get_N();
    int height_in = input_data.get_H();
    int width_in = input_data.get_W();
    int channels_in = input_data.get_C();

    int N_out = output_data.get_N();
    int height_out = output_data.get_H();
    int width_out = output_data.get_W();
    int channels_out = output_data.get_C();

    int N_filter = filter.get_N();
    int channels_filter = filter.get_C();

    assert(N_in == N_out);
    assert(channels_in==channels_filter);

    float* input_data_col = new float[kernel_h * kernel_w * channels_in *
            height_out * width_out];
    float* input_data_ptr = input_data.get_data().get();
    float* output_data_ptr = output_data.get_data().get();

    float* ones_vector = new float[height_out * width_out];
    std::fill(ones_vector, ones_vector + height_out * width_out, 1);
    for (int n = 0; n < N_in; n++) {
        im2col(input_data_ptr + n * channels_in * height_in * width_in,
               height_in, width_in, channels_in, kernel_h,
               kernel_w, pad_h, pad_w, stride_h, stride_w, input_data_col);
        gemm(CblasNoTrans, CblasNoTrans, N_filter, height_out * width_out,
             kernel_h * kernel_w * channels_in, 1, filter.get_data().get(),
             input_data_col, 0,
             output_data_ptr + n * channels_out * height_out * width_out);
        gemm(CblasNoTrans, CblasNoTrans, N_filter, height_out * width_out,
             1, 1, bias.get_data().get(), ones_vector, 1,
             output_data_ptr + n * channels_out * height_out * width_out);

    }
    delete[] ones_vector;
    delete[] input_data_col;

}

void ConvolutionLayer::backward(std::vector<Tensor>& input,
                                std::vector<Tensor>& d_input,
                                std::vector<Tensor>& d_output) {
    Tensor input_data(input[0]);
    Tensor filter(input[1]);
    Tensor d_input_data(d_input[0]);
    Tensor d_filter(d_input[1]);
    Tensor d_bias(d_input[2]);
    Tensor d_output_data(d_output[0]);

    int N_in = d_input_data.get_N();
    int height_in = d_input_data.get_H();
    int width_in = d_input_data.get_W();
    int channels_in = d_input_data.get_C();

    int N_filter = d_filter.get_N();

    int height_out = d_output_data.get_H();
    int width_out = d_output_data.get_W();
    int channels_out = d_output_data.get_C();

    float* d_input_data_ptr = d_input_data.get_data().get();
    float* d_input_data_col_ptr = new float[kernel_h * kernel_w * channels_in *
                                      height_out * width_out];
    float* input_data_ptr = input_data.get_data().get();
    float* input_data_col_ptr = new float[kernel_h * kernel_w * channels_in *
                                          height_out * width_out];
    float* d_output_date_ptr = d_output_data.get_data().get();

    float* ones_vector = new float[height_out * width_out];
    std::fill(ones_vector, ones_vector + height_out * width_out, 1);

    for (int n = 0; n < N_in; n++) {
        gemm(CblasTrans, CblasNoTrans, kernel_h * kernel_w * channels_in,
             height_out * width_out, N_filter, 1, filter.get_data().get(),
             d_output_date_ptr + n * channels_out * height_out * width_out,
             0, d_input_data_col_ptr);
        col2im(d_input_data_col_ptr, height_in, width_in, channels_in,
               kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
               d_input_data_ptr + n * channels_in * height_in * width_in);

        im2col(input_data_ptr + n * channels_in * height_in * width_in,
               height_in, width_in, channels_in, kernel_h, kernel_w,
               pad_h, pad_w, stride_h, stride_w, input_data_col_ptr);
        // accumulate the gradient each iteration
        gemm(CblasNoTrans, CblasTrans, N_filter,
             kernel_h * kernel_w * channels_in, height_out * width_out, 1,
             d_output_date_ptr + n * channels_out * height_out * width_out,
             input_data_col_ptr, 1, d_filter.get_data().get());
        gemm(CblasNoTrans, CblasNoTrans, N_filter, 1,
             height_out * width_out, 1,
             d_output_date_ptr + n * channels_out * height_out * width_out,
             ones_vector, 1, d_bias.get_data().get());
    }

    delete[] d_input_data_col_ptr;
    delete[] input_data_col_ptr;
    delete[] ones_vector;
}