# include <iostream>
# include <vector>
#include <string>
#include <fstream>
#include <cmath>
# include "Tensor.hpp"
# include "convolution_layer.hpp"

using namespace std;

void read_matrix(const char* filename, float* data) {
    string value;
    float f_value;
    int num = 0;

    //read data
    ifstream data_file(filename);
    if (!data_file.is_open()) {
        cout << filename << " NOT FOUND" << endl;
        exit(-1);
    }
    while(data_file.good()) {
        getline(data_file, value, ',');
        f_value = stof(value);
        *(data + num) = f_value;
        ++num;
    }
}

void get_sample_data(float* X, float* F, float* B, float* Y,
    float* d_X, float* d_F, float* d_B, float* d_Y) {
    read_matrix("../../test/sample_data/Conv/X.csv", X);
    read_matrix("../../test/sample_data/Conv/F.csv", F);
    read_matrix("../../test/sample_data/Conv/B.csv", B);
    read_matrix("../../test/sample_data/Conv/Y.csv", Y);
    read_matrix("../../test/sample_data/Conv/d_X.csv", d_X);
    read_matrix("../../test/sample_data/Conv/d_F.csv", d_F);
    read_matrix("../../test/sample_data/Conv/d_B.csv", d_B);
    read_matrix("../../test/sample_data/Conv/d_Y.csv", d_Y);
}

bool check_eq(float* pred, float* gndth, int size) {
    for(int iter = 0; iter < size; ++iter) {
        if(fabs(pred[iter] - gndth[iter]) /
                fabs(gndth[iter] + 1e-6) > 0.1) {
            return false;
        }
    }
    return true;
}

void test_conv_layer() {
    Tensor input_tensor(100, 32, 16, 16);
    Tensor d_input_tensor_pred(100, 32, 16, 16);
    Tensor d_input_tensor_gndth(100, 32, 16, 16);

    Tensor filter_tensor(32, 32, 5, 5);
    Tensor d_filter_tensor_pred(32, 32, 5, 5);
    Tensor d_filter_tensor_gndth(32, 32, 5, 5);

    Tensor bias_tensor(32, 1, 1, 1);
    Tensor d_bias_tensor_pred(32, 1, 1, 1);
    Tensor d_bias_tensor_gndth(32, 1, 1, 1);

    Tensor output_tensor_pred(100, 32, 8, 8);
    Tensor output_tensor_gndth(100, 32, 8, 8);
    Tensor d_output_tensor(100, 32, 8, 8);

    get_sample_data(input_tensor.get_data().get(), filter_tensor.get_data().get(),
                    bias_tensor.get_data().get(), output_tensor_gndth.get_data().get(),
                    d_input_tensor_gndth.get_data().get(), d_filter_tensor_gndth.get_data().get(),
                    d_bias_tensor_gndth.get_data().get(), d_output_tensor.get_data().get());

    int pad_h = 2, pad_w = 2, filter_num = 32, filter_channel = 32,
            kernel_h = 5, kernel_w = 5, stride_h = 2, stride_w = 2;

    ConvolutionLayer L1(pad_h, pad_w,
                        filter_num, filter_channel,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
    L1.set_filter(filter_tensor);
    L1.set_bias(bias_tensor);
    L1.set_d_filter(d_filter_tensor_pred);
    L1.set_d_bias(d_bias_tensor_pred);

    vector<Tensor> input_vector, output_vector;
    input_vector.push_back(input_tensor);
    input_vector.push_back(d_input_tensor_pred);

    output_vector.push_back(output_tensor_pred);
    output_vector.push_back(d_output_tensor);

    L1.forward(input_vector, output_vector);
    L1.backward(input_vector, output_vector);

    if (check_eq(output_tensor_pred.get_data().get(),
                 output_tensor_gndth.get_data().get(), 100*32*8*8)) {
        cout<<"test successful\n";
    }
    else {
        cout<<"test fail\n";
    }

    if (check_eq(d_input_tensor_pred.get_data().get(),
                 d_input_tensor_gndth.get_data().get(), 100*32*16*16)) {
        cout<<"test successful\n";
    }
    else {
        cout<<"test fail\n";
    }

    if (check_eq(d_filter_tensor_pred.get_data().get(),
                 d_filter_tensor_gndth.get_data().get(), 32*32*5*5)) {
        cout<<"test successful\n";
    }
    else {
        cout<<"test fail\n";
    }

    if (check_eq(d_bias_tensor_pred.get_data().get(),
                 d_bias_tensor_gndth.get_data().get(), 32)) {
        cout<<"test successful\n";
    }
    else {
        cout<<"test fail\n";
    }
}

int main() {
    test_conv_layer();
}

