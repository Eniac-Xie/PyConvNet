# include <iostream>
# include <algorithm>
# include <vector>

# include "Tensor.hpp"
# include "convolution_layer.hpp"

using namespace std;

void test_conv_layer() {
    Tensor input_tensor(2, 3, 5, 5);
    float* input_data = input_tensor.get_data();
    fill(input_data, input_data + 2*3*5*5, 1);

    Tensor d_input_tensor(2, 3, 5, 5);
    float* d_input_data = d_input_tensor.get_data();
    fill(d_input_data, d_input_data + 2*3*5*5, 1);

    Tensor filter_tensor(2, 3, 3, 3);
    float* filter = filter_tensor.get_data();
    fill(filter, filter + 2*3*3*3, 1);

    Tensor d_filter_tensor(2, 3, 3, 3);
    float* d_filter = d_filter_tensor.get_data();
    fill(d_filter, d_filter + 2*3*3*3, 1);

    Tensor bias_tensor(2, 1, 1, 1);
    float* bias = bias_tensor.get_data();
    fill(bias, bias + 2, 1);

    Tensor d_bias_tensor(2, 1, 1, 1);
    float* d_bias = d_bias_tensor.get_data();
    fill(d_bias, d_bias + 2, 0);


    Tensor output_tensor(2, 2, 3, 3);
    float* output_data = output_tensor.get_data();
    fill(output_data, output_data + 2*2*3*3, 0);

    Tensor d_output_tensor(2, 2, 3, 3);
    float* d_output_data = d_output_tensor.get_data();
    fill(d_output_data, d_output_data + 2*2*3*3, 0.5);

    int pad_h = 1, pad_w = 1, kernel_h = 3,
            kernel_w = 3, stride_h = 2,
            stride_w = 2;
    ConvolutionLayer L1(pad_h, pad_w,
                        kernel_h, kernel_w,
                        stride_h, stride_w);
    vector<Tensor*> input_vector, output_vector,
            d_input_vector, d_output_vector;
    input_vector.push_back(&input_tensor);
    input_vector.push_back(&filter_tensor);
    input_vector.push_back(&bias_tensor);

    d_input_vector.push_back(&d_input_tensor);
    d_input_vector.push_back(&d_filter_tensor);
    d_input_vector.push_back(&d_bias_tensor);

    output_vector.push_back(&output_tensor);
    d_output_vector.push_back(&d_output_tensor);


    L1.forward(input_vector, output_vector);
    L1.backward(input_vector, d_input_vector, d_output_vector);

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int k = 0; k < 3; ++k) {
                for(int l = 0; l < 3; ++l) {
                    cout<<output_tensor.get_data()[i * 18 + j * 9 + k * 3 + l]<<"\t";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            for(int k = 0; k < 5; ++k) {
                for(int l = 0; l < 5; ++l) {
                    cout<<d_input_tensor.get_data()[i * 75 + j * 25 + k * 5 + l]<<"\t";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 1; ++j) {
            for(int k = 0; k < 1; ++k) {
                for(int l = 0; l < 1; ++l) {
                    cout<<d_bias_tensor.get_data()[i * 1 + j * 1 + k * 1 + l]<<"\t";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl;
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            for(int k = 0; k < 3; ++k) {
                for(int l = 0; l < 3; ++l) {
                    cout<<d_filter_tensor.get_data()[i * 27 + j * 9 + k * 3 + l]<<"\t";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
    }
    cout<<endl;

}

int main() {
    test_conv_layer();
}

