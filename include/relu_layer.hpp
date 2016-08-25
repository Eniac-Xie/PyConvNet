#ifndef PYCONVNET_RELU_LAYER_HPP
#define PYCONVNET_RELU_LAYER_HPP

# include <vector>
# include "Tensor.hpp"
# include "Layer.hpp"

class ReLULayer: public Layer {
public:
    void params_update(float lr);
    void forward(std::vector<Tensor>& input, std::vector<Tensor>& output );
    void backward(std::vector<Tensor>& input,
                  std::vector<Tensor>& output);
};
#endif //PYCONVNET_RELU_LAYER_HPP
