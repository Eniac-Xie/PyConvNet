# include "Net.hpp"

Net::Net() {
    // empty
}

void Net::add_layer(Layer &l) {
    layers_.push_back(l);
}

void Net::forward_net() {
    for (int layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        layers_[layer_idx].forward(data_[layer_idx], data_[layer_idx + 1]);
    }
}

void Net::backward_net() {
    for (int layer_idx = layer_idx - 1; layer_idx >= 0; --layer_idx) {
        layers_[layer_idx].backward(data_[layer_idx], grads_[layer_idx], grads_[layer_idx + 1]);
    }
}