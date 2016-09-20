# include <iostream>
# include <fstream>
# include <cstring>
# include <boost/shared_array.hpp>
#include <sys/timeb.h>

# include "Layer.hpp"
# include "Net.hpp"
# include "convolution_layer.hpp"
# include "relu_layer.hpp"
# include "pooling_layer.hpp"
# include "softmax_layer.hpp"
# include "log_loss_layer.hpp"

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

float train_data[784*60000];
float train_labels[60000];
float test_data[784*10000];
float test_labels[10000];

static timeb TimingMilliSeconds;

void StartOfDuration()
{
    ftime(&TimingMilliSeconds);
}

int EndOfDuration()
{
    struct timeb now;
    ftime(&now);
    return int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) );
}

int main() {
    Net lenet;
    read_matrix("/home/xie/code/PyConvNet/data/mnist/train_image.csv", train_data);
    read_matrix("/home/xie/code/PyConvNet/data/mnist/train_label.csv", train_labels);
    read_matrix("/home/xie/code/PyConvNet/data/mnist/test_image.csv", test_data);
    read_matrix("/home/xie/code/PyConvNet/data/mnist/test_label.csv", test_labels);

    ConvolutionLayer conv1(0, 0, 20, 1, 5, 5, 1, 1);
    lenet.add_layer((Layer*)&conv1);
    lenet.add_param_layer_id(0);

    PoolingLayer pool1(0, 0, 2, 2, 2, 2);
    lenet.add_layer((Layer*)&pool1);

    ConvolutionLayer conv2(0, 0, 50, 20, 5, 5, 1, 1);
    lenet.add_layer((Layer*)&conv2);
    lenet.add_param_layer_id(2);

    PoolingLayer pool2(0, 0, 2, 2, 2, 2);
    lenet.add_layer((Layer*)&pool2);

    ConvolutionLayer conv3(0, 0, 500, 50, 4, 4, 1, 1);
    lenet.add_layer((Layer*)&conv3);
    lenet.add_param_layer_id(4);

    ReLULayer relu1;
    lenet.add_layer((Layer*)&relu1);

    ConvolutionLayer conv4(0, 0, 10, 500, 1, 1, 1, 1);
    lenet.add_layer((Layer*)&conv4);
    lenet.add_param_layer_id(6);

    SoftmaxLayer softmax1;
    lenet.add_layer((Layer*)&softmax1);

    LogLoss log1;
    lenet.add_layer((Layer*)&log1);

    int batch_size = 100;

    Tensor data0(batch_size, 1, 28, 28, 0);
    Tensor d_data0(batch_size, 1, 28, 28, 0);
    std::vector<Tensor> v0;
    v0.push_back(data0);
    v0.push_back(d_data0);

    // conv1
    Tensor data1(batch_size, 20, 24, 24, 0);
    Tensor d_data1(batch_size, 20, 24, 24, 0);
    std::vector<Tensor> v1;
    v1.push_back(data1);
    v1.push_back(d_data1);

    // pool1
    Tensor data2(batch_size, 20, 12, 12, 0);
    Tensor d_data2(batch_size, 20, 12, 12, 0);
    std::vector<Tensor> v2;
    v2.push_back(data2);
    v2.push_back(d_data2);

    // conv2
    Tensor data3(batch_size, 50, 8, 8, 0);
    Tensor d_data3(batch_size, 50, 8, 8, 0);
    std::vector<Tensor> v3;
    v3.push_back(data3);
    v3.push_back(d_data3);

    // pool2
    Tensor data4(batch_size, 50, 4, 4, 0);
    Tensor d_data4(batch_size, 50, 4, 4, 0);
    std::vector<Tensor> v4;
    v4.push_back(data4);
    v4.push_back(d_data4);

    // conv3
    Tensor data5(batch_size, 500, 1, 1, 0);
    Tensor d_data5(batch_size, 500, 1, 1, 0);
    std::vector<Tensor> v5;
    v5.push_back(data5);
    v5.push_back(d_data5);

    // relu1
    Tensor data6(batch_size, 500, 1, 1, 0);
    Tensor d_data6(batch_size, 500, 1, 1, 0);
    std::vector<Tensor> v6;
    v6.push_back(data6);
    v6.push_back(d_data6);

    // conv4
    Tensor data7(batch_size, 10, 1, 1, 0);
    Tensor d_data7(batch_size, 10, 1, 1, 0);
    std::vector<Tensor> v7;
    v7.push_back(data7);
    v7.push_back(d_data7);

    // softmax1
    Tensor data8(batch_size, 10, 1, 1, 0);
    Tensor d_data8(batch_size, 10, 1, 1, 0);
    Tensor label(batch_size, 1, 1, 1, 0);
    std::vector<Tensor> v8;
    v8.push_back(data8);
    v8.push_back(d_data8);
    v8.push_back(label);

    // log
    Tensor data9(1, 1, 1, 1, 0);
    Tensor output9(batch_size, 1, 1, 1, 0);

    std::vector<Tensor> v9;
    v9.push_back(data9);
    v9.push_back(output9);

    lenet.add_data(v0);
    lenet.add_data(v1);
    lenet.add_data(v2);
    lenet.add_data(v3);
    lenet.add_data(v4);
    lenet.add_data(v5);
    lenet.add_data(v6);
    lenet.add_data(v7);
    lenet.add_data(v8);
    lenet.add_data(v9);

    for(int i = 0; i < 1; ++i) {
        float* batch_data_ptr = new float[batch_size * 28 * 28];
        float* batch_label_ptr = new float[batch_size];
        memcpy(batch_data_ptr, train_data + i * batch_size * 28 * 28, sizeof(float) * batch_size * 28 * 28);
        vector_mul_scalar(batch_data_ptr, 255.0, batch_size * 28 * 28);
        memcpy(batch_label_ptr, train_labels + i * batch_size, sizeof(float) * batch_size);
        boost::shared_array<float> data_batch(batch_data_ptr);
        boost::shared_array<float> label_batch(batch_label_ptr);
        Tensor train_data_tensor(batch_size, 1, 28, 28, 0);
        Tensor train_label_tensor(batch_size, 1, 1, 1, 0);
        train_data_tensor.set_data(data_batch);
        train_label_tensor.set_data(label_batch);
        cout << "iter: " << i << "   ";
        StartOfDuration();
        lenet.train_batch(train_data_tensor, train_label_tensor);
        int msec = EndOfDuration();
        cout << msec << "ms elapse\n";
    }

    // do testing
    for(int i = 0; i < (int)(10000 / batch_size); ++i) {
        float* batch_data_ptr = new float[batch_size * 28 * 28];
        float* batch_label_ptr = new float[batch_size];
        memcpy(batch_data_ptr, test_data + i * batch_size * 28 * 28, sizeof(float) * batch_size * 28 * 28);
        vector_mul_scalar(batch_data_ptr, 255.0, batch_size * 28 * 28);
        memcpy(batch_label_ptr, test_labels + i * batch_size, sizeof(float) * batch_size);
        boost::shared_array<float> data_batch(batch_data_ptr);
        boost::shared_array<float> label_batch(batch_label_ptr);
        Tensor test_data_tensor(batch_size, 1, 28, 28, 0);
        Tensor test_label_tensor(batch_size, 1, 1, 1, 0);
        Tensor pred_label_tensor(batch_size, 1, 1, 1, 0);
        test_data_tensor.set_data(data_batch);
        test_label_tensor.set_data(label_batch);
        cout << "test iter: " << i << "   ";
        StartOfDuration();
        lenet.test_batch(test_data_tensor, pred_label_tensor);
        int msec = EndOfDuration();
        float* label_ptr = test_label_tensor.get_data().get();
        float* pred_ptr = pred_label_tensor.get_data().get();
        float right_num = 0.f;
        for(int j = 0; j < batch_size; ++j) {
            if (label_ptr[j] == pred_ptr[j])
                ++right_num;
        }
        cout << msec << "ms elapse, batch acc: " << right_num / batch_size << endl;
    }
}