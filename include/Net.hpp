#ifndef PYCONVNET_NET_HPP
#define PYCONVNET_NET_HPP

# include <vector>
# include "Layer.hpp"
# include "Tensor.hpp"

class Net {
public:
    Net();
    void add_layer(Layer* l);
    void add_param_layer_id(int idx);
    void add_data(std::vector<Tensor>& t);
    void forward_net(int start=-1, int end=-1);
    void backward_net();
    void params_update();
    void train_batch(Tensor& train_data, Tensor& train_label);
    void test_batch(Tensor& test_data, Tensor& pred_label);
private:
    std::vector<Layer*> layers_; // n layers
    std::vector<std::vector<Tensor>> data_; // data and d_data
    std::vector<int> params_layer_id;
    float lr;
};

#endif //PYCONVNET_NET_HPP
