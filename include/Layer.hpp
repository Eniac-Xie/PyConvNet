#ifndef PYCONVNET_LAYER_HPP
#define PYCONVNET_LAYER_HPP

# include "Tensor.hpp"

class Layer {
public:
    Layer() {
        // empty
    }

    virtual void forward(std::vector<Tensor>& input, std::vector<Tensor>& output)=0;

    virtual void backward(std::vector<Tensor>& input,
                  std::vector<Tensor>& d_input,
                  std::vector<Tensor>& d_output)=0;
};

#endif //PYCONVNET_LAYER_HPP
