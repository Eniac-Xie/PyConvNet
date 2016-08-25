#ifndef PYCONVNET_SOFTMAX_LAYER_HPP
#define PYCONVNET_SOFTMAX_LAYER_HPP

# include "Layer.hpp"

class SoftmaxLayer: public Layer {
public:
    void params_update(float lr);
    void forward(std::vector<Tensor>& input, std::vector<Tensor>& output);
    void backward(std::vector<Tensor>& input,
                  std::vector<Tensor>& output);
};
#endif //PYCONVNET_SOFTMAX_LAYER_HPP
