#include <fstream>
#include <cmath>
#include <vector>
#include "Tensor.hpp"
#include "relu_layer.hpp"

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

void get_sample_data(float* X, float* Y,
                     float* d_X, float* d_Y) {
    read_matrix("../../test/sample_data/ReLU/X.csv", X);
    read_matrix("../../test/sample_data/ReLU/Y.csv", Y);
    read_matrix("../../test/sample_data/ReLU/d_X.csv", d_X);
    read_matrix("../../test/sample_data/ReLU/d_Y.csv", d_Y);
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

void test_pooling_layer() {
    Tensor input_tensor(100, 32, 16, 16);
    Tensor d_input_tensor_pred(100, 32, 16, 16);
    Tensor d_input_tensor_gndth(100, 32, 16, 16);

    Tensor output_tensor_pred(100, 32, 16, 16);
    Tensor output_tensor_gndth(100, 32, 16, 16);
    Tensor d_output_tensor(100, 32, 16, 16);

    get_sample_data(input_tensor.get_data().get(), output_tensor_gndth.get_data().get(),
                    d_input_tensor_gndth.get_data().get(), d_output_tensor.get_data().get());

    ReLULayer L1;
    vector<Tensor> input_vector, output_vector,
            d_input_vector, d_output_vector;

    input_vector.push_back(input_tensor);
    d_input_vector.push_back(d_input_tensor_pred);

    output_vector.push_back(output_tensor_pred);
    d_output_vector.push_back(d_output_tensor);

    L1.forward(input_vector, output_vector);
    L1.backward(input_vector, d_input_vector, d_output_vector);

    if (check_eq(output_tensor_pred.get_data().get(),
                 output_tensor_gndth.get_data().get(), 100*32*16*16)) {
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
}

int main() {
    test_pooling_layer();
}