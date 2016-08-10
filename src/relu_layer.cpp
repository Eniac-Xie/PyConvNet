# include "relu_layer.hpp"

void ReLULayer::forward(std::vector<Tensor> &input, std::vector<Tensor> &output) {
    Tensor input_data(input[0]);
    Tensor output_data(output[0]);

    float* input_data_ptr = input_data.get_data().get();
    float* output_data_ptr = output_data.get_data().get();

    int N_in = input_data.get_N();
    int C_in = input_data.get_C();
    int H_in = input_data.get_H();
    int W_in = input_data.get_W();
    int data_size = N_in * C_in * H_in * W_in;
    for(int index = 0; index < data_size; ++index) {
        output_data_ptr[index] = input_data_ptr[index] > 0 ? input_data_ptr[index] : 0;
    }
}

void ReLULayer::backward(std::vector<Tensor> &input, std::vector<Tensor> &d_input,
                         std::vector<Tensor> &d_output) {
    Tensor input_data(input[0]);
    Tensor d_input_data(d_input[0]);
    Tensor d_output_data(d_output[0]);

    float* input_data_ptr = input_data.get_data().get();
    float* d_input_data_ptr = d_input_data.get_data().get();
    float* d_output_data_ptr = d_output_data.get_data().get();

    int N_in = d_input_data.get_N();
    int C_in = d_input_data.get_C();
    int H_in = d_input_data.get_H();
    int W_in = d_input_data.get_W();
    int data_size = d_input_data.get_size();

    for(int index = 0; index < data_size; ++index) {
        d_input_data_ptr[index] = input_data_ptr[index] > 0 ? d_output_data_ptr[index] : 0;
    }
}

