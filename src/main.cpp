# include <iostream>
# include <vector>

# include "Tensor.hpp"
# include "convolution_layer.hpp"

using namespace std;

int main() {
    Tensor data(2, 3, 5, 5);
    float* data_ptr = data.get_data();
    for (int i = 0; i < 2*3*5*5; ++i)
        data_ptr[i] = 1;

    Tensor filter(2, 3, 3, 3);
    float* filter_ptr = filter.get_data();
    for (int i = 0; i < 2*3*3*3; ++i)
        filter_ptr[i] = 1;

    Tensor d_data(2, 3, 5, 5);
    float* d_data_ptr = d_data.get_data();
    for (int i = 0; i < 2*3*5*5; ++i)
        d_data_ptr[i] = 1;

    Tensor d_filter(2, 3, 3, 3);
    float* d_filter_ptr = d_filter.get_data();
    for (int i = 0; i < 2*3*3*3; ++i)
        d_filter_ptr[i] = 1;

    Tensor output_data(2, 2, 3, 3);
    float* output_data_ptr = output_data.get_data();
    for (int i = 0; i < 2*2*3*3; ++i)
        output_data_ptr[i] = 1;

    vector<Tensor*> input, d_input, output;
    input.push_back(&data);
    input.push_back(&filter);
    d_input.push_back(&d_data);
    d_input.push_back(&d_filter);

    output.push_back(&output_data);
    ConvolutionLayer l1(1, 1, 3, 3, 2, 2);
    l1.forward(input, output);
    l1.backward(input, d_input,output);
    for (int i = 0; i < 2*2*3*3; ++i) {
        cout << output[0]->get_data()[i] << '\t';
    }
    cout<<endl;
    filter_ptr = d_filter.get_data();
    for (int i = 0; i < 2*3*3*3; ++i)
        cout << filter_ptr[i] << '\t';
    cout<<endl;
    filter_ptr = d_data.get_data();
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 5; ++k) {
                for(int l = 0; l < 5; ++l) {
                    cout << filter_ptr[i*3*5*5 + j*5*5 + k*5 + l] << '\t';
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout<<endl;
    }

}
