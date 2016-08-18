#ifndef PYCONVNET_NET_HPP
#define PYCONVNET_NET_HPP

# include <vector>
# include "Layer.hpp"
# include "Tensor.hpp"

class Net {
public:
    Net();
    void add_layer(Layer* l);
    void forward_net();
    void backward_net();
    void params_update();
    void train_batch(Tensor train_data, Tensor train_label);
private:
    std::vector<Layer*> layers_; // n layers
    std::vector<std::vector<Tensor>> data_; // n+1
    std::vector<std::vector<Tensor>> grads_; // n+1
    std::vector<int> params_layer_id;
    float lr;
    float loss;
};

#endif //PYCONVNET_NET_HPP
