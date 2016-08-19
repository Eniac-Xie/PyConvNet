#ifndef PYCONVNET_SOFTMAX_LOG_LOSS_LAYER_HPP
#define PYCONVNET_SOFTMAX_LOG_LOSS_LAYER_HPP

# include "Layer.hpp"

class LogLoss: public Layer {
public:
    void forward(std::vector<Tensor>& input, std::vector<Tensor>& output);
    void backward(std::vector<Tensor> &input,
                           std::vector<Tensor> &output);
};
#endif //PYCONVNET_SOFTMAX_LOG_LOSS_LAYER_HPP
