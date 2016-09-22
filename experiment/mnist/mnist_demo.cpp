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

int reverse_int (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i&255;
    ch2 = (i >> 8)&255;
    ch3 = (i >> 16)&255;
    ch4 = (i >> 24)&255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_mnist(const char* data_path, const char* label_path, float* data, float* label) {
    std::ifstream data_file (data_path, std::ios::binary);
    if (data_file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        data_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number= reverse_int(magic_number);
        data_file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        data_file.read((char*)&n_rows, sizeof(n_rows));
        n_rows= reverse_int(n_rows);
        data_file.read((char*)&n_cols, sizeof(n_cols));
        n_cols= reverse_int(n_cols);

        for(int i = 0; i < number_of_images; ++i)
        {
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp=0;
                    data_file.read((char*)&temp,sizeof(temp));
                    data[i * n_rows * n_cols + r * n_cols + c]= (float)temp;
                }
            }
        }
    }
    data_file.close();

    std::ifstream label_file (label_path, std::ios::binary);
    if (label_file.is_open())
    {
        int magic_number = 0;
        int number_of_items = 0;
        label_file.read((char*)&magic_number, sizeof(magic_number));
        magic_number= reverse_int(magic_number);
        label_file.read((char*)&number_of_items, sizeof(number_of_items));
        number_of_items = reverse_int(number_of_items);

        for(int i = 0; i < number_of_items; ++i)
        {
            unsigned char temp=0;
            label_file.read((char*)&temp,sizeof(temp));
            label[i]= (float)temp;
        }
    }
    label_file.close();
}

float train_data[784*60000];
float train_label[60000];
float test_data[784*10000];
float test_label[10000];

static timeb TimingMilliSeconds;

void StartOfDuration() {
    ftime(&TimingMilliSeconds);
}

int EndOfDuration() {
    struct timeb now;
    ftime(&now);
    return int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) );
}

int main() {
    // loading mnist data
    read_mnist("../../data/mnist/train-images-idx3-ubyte",
               "../../data/mnist/train-labels-idx1-ubyte",
               train_data, train_label);
    read_mnist("../../data/mnist/t10k-images-idx3-ubyte",
               "../../data/mnist/t10k-labels-idx1-ubyte",
               test_data, test_label);

    Net lenet;
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

    Tensor data0(batch_size, 1, 28, 28);
    Tensor d_data0(batch_size, 1, 28, 28);
    std::vector<Tensor> v0;
    v0.push_back(data0);
    v0.push_back(d_data0);

    // conv1
    Tensor data1(batch_size, 20, 24, 24);
    Tensor d_data1(batch_size, 20, 24, 24);
    std::vector<Tensor> v1;
    v1.push_back(data1);
    v1.push_back(d_data1);

    // pool1
    Tensor data2(batch_size, 20, 12, 12);
    Tensor d_data2(batch_size, 20, 12, 12);
    std::vector<Tensor> v2;
    v2.push_back(data2);
    v2.push_back(d_data2);

    // conv2
    Tensor data3(batch_size, 50, 8, 8);
    Tensor d_data3(batch_size, 50, 8, 8);
    std::vector<Tensor> v3;
    v3.push_back(data3);
    v3.push_back(d_data3);

    // pool2
    Tensor data4(batch_size, 50, 4, 4);
    Tensor d_data4(batch_size, 50, 4, 4);
    std::vector<Tensor> v4;
    v4.push_back(data4);
    v4.push_back(d_data4);

    // conv3
    Tensor data5(batch_size, 500, 1, 1);
    Tensor d_data5(batch_size, 500, 1, 1);
    std::vector<Tensor> v5;
    v5.push_back(data5);
    v5.push_back(d_data5);

    // relu1
    Tensor data6(batch_size, 500, 1, 1);
    Tensor d_data6(batch_size, 500, 1, 1);
    std::vector<Tensor> v6;
    v6.push_back(data6);
    v6.push_back(d_data6);

    // conv4
    Tensor data7(batch_size, 10, 1, 1);
    Tensor d_data7(batch_size, 10, 1, 1);
    std::vector<Tensor> v7;
    v7.push_back(data7);
    v7.push_back(d_data7);

    // softmax1
    Tensor data8(batch_size, 10, 1, 1);
    Tensor d_data8(batch_size, 10, 1, 1);
    Tensor label(batch_size, 1, 1, 1);
    std::vector<Tensor> v8;
    v8.push_back(data8);
    v8.push_back(d_data8);
    v8.push_back(label);

    // log
    Tensor data9(1, 1, 1, 1);
    Tensor output9(batch_size, 1, 1, 1);

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
    
    int epochs = 5;
    float lr_arr[5] = {0.0001, 0.0001, 0.00001, 0.00001, 0.00001};
    for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx) {
        lenet.set_lr(lr_arr[epoch_idx]);
        for(int i = 0; i < (int)(60000 / batch_size); ++i) {
            float* batch_data_ptr = new float[batch_size * 28 * 28];
            float* batch_label_ptr = new float[batch_size];
            memcpy(batch_data_ptr, train_data + i * batch_size * 28 * 28, sizeof(float) * batch_size * 28 * 28);
            memcpy(batch_label_ptr, train_label + i * batch_size, sizeof(float) * batch_size);
            boost::shared_array<float> data_batch(batch_data_ptr);
            boost::shared_array<float> label_batch(batch_label_ptr);
            Tensor train_data_tensor(batch_size, 1, 28, 28);
            Tensor train_label_tensor(batch_size, 1, 1, 1);
            train_data_tensor.set_data(data_batch);
            train_label_tensor.set_data(label_batch);
            std::cout << "epoch: " << epoch_idx + 1 << ", iter: " << i << ", ";
            StartOfDuration();
            lenet.train_batch(train_data_tensor, train_label_tensor);
            int msec = EndOfDuration();
            std::cout << msec << "ms elapse\n";
        }
    }

    // do testing
    float right_num = 0.f;
    for(int i = 0; i < (int)(10000 / batch_size); ++i) {
        float* batch_data_ptr = new float[batch_size * 28 * 28];
        float* batch_label_ptr = new float[batch_size];
        memcpy(batch_data_ptr, test_data + i * batch_size * 28 * 28, sizeof(float) * batch_size * 28 * 28);
        memcpy(batch_label_ptr, test_label + i * batch_size, sizeof(float) * batch_size);
        boost::shared_array<float> data_batch(batch_data_ptr);
        boost::shared_array<float> label_batch(batch_label_ptr);
        Tensor test_data_tensor(batch_size, 1, 28, 28);
        Tensor test_label_tensor(batch_size, 1, 1, 1);
        Tensor pred_label_tensor(batch_size, 1, 1, 1);
        test_data_tensor.set_data(data_batch);
        test_label_tensor.set_data(label_batch);
        std::cout << "test iter: " << i << "   ";
        StartOfDuration();
        lenet.test_batch(test_data_tensor, pred_label_tensor);
        int msec = EndOfDuration();
        float* label_ptr = test_label_tensor.get_data().get();
        float* pred_ptr = pred_label_tensor.get_data().get();
        for(int j = 0; j < batch_size; ++j) {
            if (label_ptr[j] == pred_ptr[j])
                ++right_num;
        }
        std::cout << msec << "ms elapse, batch acc: " << right_num / ((i+1) * batch_size) << std::endl;
    }
}
