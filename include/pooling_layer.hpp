#ifndef PYCONVNET_POOLING_LAYER_HPP
#define PYCONVNET_POOLING_LAYER_HPP

# include <Tensor.hpp>
# include <vector>

class PoolingLayer {
public:
    PoolingLayer(const int pad_h_, const int pad_w_,
                 const int kernel_h_, const int kernel_w_,
                 const int stride_h_, const int stride_w_) {
        pad_h = pad_h_;
        pad_w = pad_w_;
        kernel_h = kernel_h_;
        kernel_w = kernel_w_;
        stride_h = stride_h_;
        stride_w = stride_w_;
        max_index_mask.clear();
    };

    void forward(std::vector<Tensor>& input, std::vector<Tensor>& output );
    void backward(std::vector<Tensor>& input,
                  std::vector<Tensor>& d_input,
                  std::vector<Tensor>& d_output);

private:
    int pad_h = 0;
    int pad_w = 0;
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 0;
    int stride_w = 0;
    std::vector<int> max_index_mask;
};
#endif //PYCONVNET_POOLING_LAYER_HPP
