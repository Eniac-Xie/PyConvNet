#ifndef PYCONVNET_MNIST_DEMO_HPP
#define PYCONVNET_MNIST_DEMO_HPP

# include "Layer.hpp"
# include "Net.hpp"
# include "convolution_layer.hpp"
# include "relu_layer.hpp"
# include "pooling_layer.hpp"
# include "softmax_layer.hpp"
# include "log_loss_layer.hpp"

// function to read mnist data
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

std::shared_ptr<Net> init_lenet(int batch_size) {
    std::shared_ptr<Net> lenet(new Net());
    std::shared_ptr<ConvolutionLayer> conv1(new ConvolutionLayer(0, 0, 20, 1, 5, 5, 1, 1));
    lenet->add_layer(conv1);
    lenet->add_param_layer_id(0);

    std::shared_ptr<PoolingLayer> pool1(new PoolingLayer(0, 0, 2, 2, 2, 2));
    lenet->add_layer(pool1);

    std::shared_ptr<ConvolutionLayer> conv2(new ConvolutionLayer(0, 0, 50, 20, 5, 5, 1, 1));
    lenet->add_layer(conv2);
    lenet->add_param_layer_id(2);

    std::shared_ptr<PoolingLayer> pool2(new PoolingLayer(0, 0, 2, 2, 2, 2));
    lenet->add_layer(pool2);

    std::shared_ptr<ConvolutionLayer> conv3(new ConvolutionLayer(0, 0, 500, 50, 4, 4, 1, 1));
    lenet->add_layer(conv3);
    lenet->add_param_layer_id(4);

    std::shared_ptr<ReLULayer> relu1(new ReLULayer());
    lenet->add_layer(relu1);

    std::shared_ptr<ConvolutionLayer> conv4(new ConvolutionLayer(0, 0, 10, 500, 1, 1, 1, 1));
    lenet->add_layer(conv4);
    lenet->add_param_layer_id(6);

    std::shared_ptr<SoftmaxLayer> softmax1(new SoftmaxLayer());
    lenet->add_layer(softmax1);

    std::shared_ptr<LogLoss> log1(new LogLoss());
    lenet->add_layer(log1);



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

    lenet->add_data(v0);
    lenet->add_data(v1);
    lenet->add_data(v2);
    lenet->add_data(v3);
    lenet->add_data(v4);
    lenet->add_data(v5);
    lenet->add_data(v6);
    lenet->add_data(v7);
    lenet->add_data(v8);
    lenet->add_data(v9);
    return lenet;
}

#endif //PYCONVNET_MNIST_DEMO_HPP
