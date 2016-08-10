# include <assert.h>
# include <numeric>
# include <algorithm>

# include "blas_function.hpp"
# include "softmax_layer.hpp"
# include "Tensor.hpp"

void SoftmaxLayer::forward(std::vector <Tensor> &input, std::vector <Tensor> &output) {
    Tensor input_data(input[0]);
    Tensor output_data(output[0]);

    float* input_data_ptr = input_data.get_data().get();
    float* output_data_ptr = output_data.get_data().get();

    int N_in = input_data.get_N();
    int C_in = input_data.get_C();
    int H_in = input_data.get_H();
    int W_in = input_data.get_W();
    int size_in = C_in * H_in * W_in;

    assert(W_in == 1 && H_in == 1);

    for( int n = 0; n < N_in; ++n) {
        float max_ele = *(std::max_element(input_data_ptr + n * size_in,
                                           input_data_ptr + (n + 1) * size_in));
        vector_sub_scalar(input_data_ptr + n * size_in, max_ele,
                          output_data_ptr + n * size_in, size_in);

        vector_exp(output_data_ptr + n * size_in,
                   output_data_ptr + n * size_in, size_in);

        float sum_exp = vector_sum(output_data_ptr + n * size_in, size_in);

        vector_div_scalar(output_data_ptr + n * size_in,
                          sum_exp, size_in);
    }
}

void SoftmaxLayer::backward(std::vector<Tensor> &output,
                            std::vector<Tensor> &d_input,
                            std::vector<Tensor> &d_output) {
    Tensor output_data(output[0]);
    Tensor d_input_data(d_input[0]);
    Tensor d_output_data(d_output[0]);

    float* output_data_ptr = output_data.get_data().get();
    float* d_input_data_ptr = d_input_data.get_data().get();
    float* d_output_data_ptr = d_output_data.get_data().get();

    int N_in = d_input_data.get_N();
    int C_in = d_input_data.get_C();
    int H_in = d_input_data.get_H();
    int W_in = d_input_data.get_W();
    int data_size = C_in * H_in * W_in;

    assert(W_in == 1 && H_in == 1);

    for(int n = 0; n < N_in; ++n) {
        vector_mul(d_output_data_ptr + n * data_size,
                   output_data_ptr + n * data_size,
                   d_input_data_ptr + n * data_size, data_size);
        float dy_dot_y = 0;
        gemm(CblasNoTrans, CblasNoTrans, 1, 1, data_size, 1,
             d_output_data_ptr + n * data_size,
             output_data_ptr + n * data_size, 0, &dy_dot_y);
        gemm(CblasNoTrans, CblasNoTrans, 1, data_size, 1, -1,
             &dy_dot_y, output_data_ptr + n * data_size, 1,
             d_input_data_ptr + n * data_size);
    }
}