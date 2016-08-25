# include <algorithm>
# include <cmath>
# include <iostream>

# include "log_loss_layer.hpp"

void LogLoss::params_update(float lr) {
    // empty
}
void LogLoss::forward(std::vector <Tensor> &input, std::vector <Tensor> &output) {
    Tensor input_data(input[0]);
    Tensor label(input[2]);
    Tensor loss(output[0]);
    Tensor output_data(output[1]);


    float* input_data_ptr = input_data.get_data().get();
    float* label_ptr = label.get_data().get();
    float* output_data_ptr = output_data.get_data().get();
    float* loss_ptr = loss.get_data().get();

    loss_ptr[0] = 0;

    int N_in = input_data.get_N();
    int C_in = input_data.get_C();
    int H_in = input_data.get_H();
    int W_in = input_data.get_W();
    int size_in = C_in * H_in * W_in;

    assert(W_in == 1 && H_in == 1);

    for( int n = 0; n < N_in; ++n) {
        float gndth_prob = -logf(*(input_data_ptr + n * size_in + (int)(label_ptr[n])));
        output_data_ptr[n] = *(input_data_ptr + n * size_in + (int)(label_ptr[n]));
        loss_ptr[0] += gndth_prob;
    }
}

void LogLoss::backward(std::vector<Tensor> &input,
                       std::vector<Tensor> &output) {
    Tensor output_data(output[1]);
    Tensor label(input[2]);
    Tensor d_input_data(input[1]);

    float* output_data_ptr = output_data.get_data().get();
    float* label_ptr = label.get_data().get();
    float* d_input_data_ptr = d_input_data.get_data().get();

    int N_in = d_input_data.get_N();
    int C_in = d_input_data.get_C();
    int H_in = d_input_data.get_H();
    int W_in = d_input_data.get_W();
    int data_size = C_in * H_in * W_in;

    assert(W_in == 1 && H_in == 1);
    std::fill_n(d_input_data_ptr, d_input_data.get_size(), 0);
    for(int n = 0; n < N_in; ++n) {
        *(d_input_data_ptr + n * data_size + (int)(label_ptr[n])) = -1 / output_data_ptr[n];
    }
}