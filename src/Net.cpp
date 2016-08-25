# include <iostream>
# include <vector>

# include "Net.hpp"
//# include "convolution_layer.hpp"
//# include "relu_layer.hpp"
//# include "pooling_layer.hpp"
//# include "softmax_layer.hpp"
//# include "log_loss_layer.hpp"

Net::Net() {
    this->lr = 0.01;
//    ConvolutionLayer conv1(0, 0, 20, 1, 5, 5, 1, 1);
//    this->layers_.push_back(&conv1);
//
//    PoolingLayer pool1(0, 0, 2, 2, 2, 2);
//    this->layers_.push_back(&pool1);
//
//    ConvolutionLayer conv2(0, 0, 50, 20, 5, 5, 1, 1);
//    this->layers_.push_back(&conv2);
//
//    PoolingLayer pool2(0, 0, 2, 2, 2, 2);
//    this->layers_.push_back(&pool2);
//
//    ConvolutionLayer conv3(0, 0, 500, 50, 4, 4, 1, 1);
//    this->layers_.push_back(&conv3);
//
//    ReLULayer relu1;
//    this->layers_.push_back(&relu1);
//
//    ConvolutionLayer conv4(0, 0, 500, 10, 1, 1, 1, 1);
//    this->layers_.push_back(&conv4);
//
//    SoftmaxLayer softmax1;
//    this->layers_.push_back(&softmax1);
//
//    LogLoss log1;
//    this->layers_.push_back(&log1);

//    int batch_size = 10;
//
//    Tensor data0(batch_size, 1, 28, 28);
//    Tensor d_data0(batch_size, 1, 28, 28);
//    std::vector<Tensor> v0;
//    v0.push_back(data0);
//    v0.push_back(d_data0);
//
//    // conv1
//    Tensor data1(batch_size, 20, 24, 24);
//    Tensor d_data1(batch_size, 20, 24, 24);
//    std::vector<Tensor> v1;
//    v1.push_back(data1);
//    v1.push_back(d_data1);
//
//    // pool1
//    Tensor data2(batch_size, 20, 12, 12);
//    Tensor d_data2(batch_size, 20, 12, 12);
//    std::vector<Tensor> v2;
//    v2.push_back(data2);
//    v2.push_back(d_data2);
//
//    // conv2
//    Tensor data3(batch_size, 50, 8, 8);
//    Tensor d_data3(batch_size, 50, 8, 8);
//    std::vector<Tensor> v3;
//    v3.push_back(data3);
//    v3.push_back(d_data3);
//
//    // pool2
//    Tensor data4(batch_size, 50, 4, 4);
//    Tensor d_data4(batch_size, 50, 4, 4);
//    std::vector<Tensor> v4;
//    v4.push_back(data4);
//    v4.push_back(d_data4);
//
//    // conv3
//    Tensor data5(batch_size, 500, 1, 1);
//    Tensor d_data5(batch_size, 500, 1, 1);
//    std::vector<Tensor> v5;
//    v5.push_back(data5);
//    v5.push_back(d_data5);
//
//    // relu1
//    Tensor data6(batch_size, 500, 1, 1);
//    Tensor d_data6(batch_size, 500, 1, 1);
//    std::vector<Tensor> v6;
//    v6.push_back(data6);
//    v6.push_back(d_data6);
//
//    // conv4
//    Tensor data7(batch_size, 10, 1, 1);
//    Tensor d_data7(batch_size, 10, 1, 1);
//    std::vector<Tensor> v7;
//    v7.push_back(data7);
//    v7.push_back(d_data7);
//
//    // softmax1
//    Tensor data8(batch_size, 10, 1, 1);
//    Tensor d_data8(batch_size, 10, 1, 1);
//    Tensor label(batch_size, 1, 1, 1);
//    std::vector<Tensor> v8;
//    v8.push_back(data8);
//    v8.push_back(d_data8);
//    v8.push_back(label);
//
//    // log
//    Tensor data9(1, 1, 1, 1);
//    Tensor d_data9(batch_size, 1, 1, 1);
//
//    std::vector<Tensor> v9;
//    v9.push_back(data9);
//    v9.push_back(d_data9);
//
//    this->data_.push_back(v0);
//    this->data_.push_back(v1);
//    this->data_.push_back(v2);
//    this->data_.push_back(v3);
//    this->data_.push_back(v4);
//    this->data_.push_back(v5);
//    this->data_.push_back(v6);
//    this->data_.push_back(v7);
//    this->data_.push_back(v8);
//    this->data_.push_back(v9);

}

void Net::add_layer(Layer* l) {
    layers_.push_back(l);
}

void Net::add_param_layer_id(int idx) {
    this->params_layer_id.push_back(idx);
}

void Net::add_data(std::vector<Tensor>& t) {
    this->data_.push_back(t);
}

void Net::forward_net() {
    for (int layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        layers_[layer_idx]->forward(data_[layer_idx], data_[layer_idx + 1]);
    }
}

void Net::backward_net() {
    for (int layer_idx = layers_.size() - 1; layer_idx >= 0; --layer_idx) {
        layers_[layer_idx]->backward(data_[layer_idx], data_[layer_idx + 1]);
    }
}

void Net::params_update() {
    for(int idx = 0; idx < params_layer_id.size(); ++idx) {
        this->layers_[params_layer_id[idx]]->params_update(this->lr);
    }
}

void Net::train_batch(Tensor& train_data, Tensor& train_label) {
    // TODO
    // implement "=" operator of class Tensor
    this->data_[0][0] = train_data;
    this->data_[this->data_.size() - 2][2] = train_label;
    this->forward_net();
    std::cout << "loss: " << *(data_[this->data_.size() - 1][0].get_data().get())
            << std::endl;
    this->backward_net();
    this->params_update();
}