# include <iostream>
# include <vector>

# include "Net.hpp"

Net::Net() {
    this->lr = 0.0001;
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

void Net::forward_net(int start, int end) {

    for (int layer_idx = start > 0 ? start:0;
         layer_idx < (end > 0 ? end:layers_.size()); ++layer_idx) {
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
            << "   ";
    this->backward_net();
    this->params_update();
}

void Net::test_batch(Tensor &test_data, Tensor& pred_label) {
    this->data_[0][0] = test_data;
    this->forward_net(this->layers_.size() - 2);
    Tensor result = this->data_[this->data_.size() - 2][0];
    int batch_size = result.get_N();
    int channel_size = result.get_C();
    assert(result.get_H() == 1 && result.get_W() == 1);

    float* res_ptr = result.get_data().get();
    float* pred_label_ptr = pred_label.get_data().get();
    for (int i = 0; i < batch_size; i++) {
        pred_label_ptr[i] = std::distance(res_ptr + channel_size * i,
                      std::max_element(res_ptr + channel_size * i,
                                       res_ptr + channel_size * (i + 1)));
    }
}
