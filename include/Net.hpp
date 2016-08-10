#ifndef PYCONVNET_NET_HPP
#define PYCONVNET_NET_HPP

# include <vector>
# include "Layer.hpp"
# include "Tensor.hpp"

class Net {
public:
    Net();
    void add_layer(Layer& l);
    void forward_net();
    void backward_net();
    void train();
private:
    std::vector<Layer> layers_;
    std::vector<std::vector<Tensor>> data_;
    std::vector<std::vector<Tensor>> grads_;
    float lr;
    float loss;
};

#endif //PYCONVNET_NET_HPP
