#ifndef PYCONVNET_RELU_LAYER_HPP
#define PYCONVNET_RELU_LAYER_HPP

# include <vector>
# include "Tensor.hpp"

class ReLULayer {
public:
    ReLULayer() {
        // empty
    }
    void forward(std::vector<Tensor>& input, std::vector<Tensor>& output );
    void backward(std::vector<Tensor>& input,
                  std::vector<Tensor>& d_input,
                  std::vector<Tensor>& d_output);
};
#endif //PYCONVNET_RELU_LAYER_HPP
