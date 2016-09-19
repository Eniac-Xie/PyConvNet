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
            << "   ";
    this->backward_net();
    this->params_update();
}
