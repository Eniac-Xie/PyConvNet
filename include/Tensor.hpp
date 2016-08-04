//
// Created by xie on 16-8-3.
//

#ifndef PYCONVNET_TENSOR_HPP
#define PYCONVNET_TENSOR_HPP

# include <iostream>
class Tensor {
public:
    Tensor(int n, int c, int h, int w) {
        N = n;
        C = c;
        H = h;
        W = w;
        data = new float[N * C * H * W];
    }
    Tensor(const Tensor& T) {
        std::cout<<"Copy Tensor\n";
        N = T.get_N();
        C = T.get_C();
        H = T.get_H();
        W = T.get_W();
        data = T.get_data();
    }
    float*  get_data() const {
        return data;
    }
    int  get_N() const {
        return N;
    }
    int  get_C() const {
        return C;
    }
    int  get_H() const {
        return H;
    }
    int  get_W() const {
        return W;
    }
    ~Tensor() {
        delete[] data;
    }
private:
    float* data;
    int N, C, H, W;
};
#endif //PYCONVNET_TENSOR_HPP
