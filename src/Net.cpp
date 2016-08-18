# include "Net.hpp"
# include "convolution_layer.hpp"

Net::Net() {
    // empty
    loss = 0;
    float* loss_grads_ptr = data_grads_[n].get_data().get();
    *loss_grads_ptr = 1;
}

void Net::add_layer(Layer* l) {
    layers_.push_back(l);
}

void Net::forward_net() {
    for (int layer_idx = 0; layer_idx < layers_.size(); ++layer_idx) {
        layers_[layer_idx]->forward(data_[layer_idx], data_[layer_idx + 1]);
    }
}

void Net::backward_net() {
    layers_[layers_.size() - 1]->backward(data_[layers_.size() - 1],
                                 data_[layers_.size()], grads_[layers_.size() - 1]);
    layers_[layers_.size() - 2]->backward(data_[layers_.size() - 1],
                                          grads_[layers_.size() - 2], grads_[layers_.size() - 1]);
    for (int layer_idx = layers_.size() - 3; layer_idx >= 0; --layer_idx) {
        layers_[layer_idx]->backward(data_[layer_idx], grads_[layer_idx],
                                     grads_[layer_idx + 1]);
    }
}

void Net::params_update() {
    for(int idx = 0; idx < params_layer_id.size(); ++idx) {
        Tensor filter(data_[params_layer_id[idx]][1]);
        Tensor bias(data_[params_layer_id[idx]][2]);

        Tensor d_filter(grads_[params_layer_id[idx]][1]);
        Tensor d_bias(grads_[params_layer_id[idx]][2]);

        // update params
        filter = filter - d_filter * this->lr;
        bias = bias - d_bias * (2 * this->lr);
    }
}

void Net::train_batch(Tensor train_data, Tensor train_label) {
    
}