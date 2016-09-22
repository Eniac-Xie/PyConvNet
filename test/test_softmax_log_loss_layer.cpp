#include <fstream>
#include <cmath>
#include <vector>
#include "Tensor.hpp"
#include "softmax_layer.hpp"
#include "log_loss_layer.hpp"

using namespace std;

void read_matrix(const char* filename, float* data, float coeff=1) {
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
        *(data + num) = f_value * coeff;
        ++num;
    }
}

void get_sample_data(float* X, float* L, float* Y, float* d_X) {
    read_matrix("../../test/sample_data/SoftMaxLogLoss/X.csv", X);
    read_matrix("../../test/sample_data/SoftMaxLogLoss/L.csv", L);
    // the test data is unnormalized, so we need add a coeff to normalize loss and grads
    read_matrix("../../test/sample_data/SoftMaxLogLoss/Y.csv", Y, 0.01);
    read_matrix("../../test/sample_data/SoftMaxLogLoss/d_X.csv", d_X, 0.01);
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

void test_softmax_log_loss_layer() {
    Tensor input_tensor(100, 10, 1, 1);
    Tensor label_tensor(100, 1, 1, 1);
    Tensor d_input_tensor_pred(100, 10, 1, 1);
    Tensor d_input_tensor_gndth(100, 10, 1, 1);

    Tensor intermediate_tensor(100, 10, 1, 1);
    Tensor d_intermediate_tensor(100, 10, 1, 1);

    Tensor output_tensor_pred(100, 1, 1, 1);

    Tensor loss_pred(1, 1, 1, 1);
    Tensor loss_gndth(1, 1, 1, 1);

    get_sample_data(input_tensor.get_data().get(), label_tensor.get_data().get(),
                    loss_gndth.get_data().get(),
                    d_input_tensor_gndth.get_data().get());

    SoftmaxLayer L1;
    LogLoss L2;

    vector<Tensor> input_vector, intermediate_vector, output_vector;

    input_vector.push_back(input_tensor);
    input_vector.push_back(d_input_tensor_pred);

    intermediate_vector.push_back(intermediate_tensor);
    intermediate_vector.push_back(d_intermediate_tensor);
    intermediate_vector.push_back(label_tensor);

    output_vector.push_back(loss_pred);
    output_vector.push_back(output_tensor_pred);

    L1.forward(input_vector, intermediate_vector);
    L2.forward(intermediate_vector, output_vector);
    L2.backward(intermediate_vector, output_vector);
    L1.backward(input_vector, intermediate_vector);

    if (check_eq(loss_pred.get_data().get(),
                 loss_gndth.get_data().get(), 1)) {
        cout<<"test successful\n";
    }
    else {
        cout<<"test fail\n";
    }

    if (check_eq(d_input_tensor_pred.get_data().get(),
                 d_input_tensor_gndth.get_data().get(), 100*10*1*1)) {
        cout<<"test successful\n";
    }
    else {
        cout<<"test fail\n";
    }
}

int main() {
    test_softmax_log_loss_layer();
}