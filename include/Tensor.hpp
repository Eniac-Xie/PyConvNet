#ifndef PYCONVNET_TENSOR_HPP
#define PYCONVNET_TENSOR_HPP

# include <iostream>
# include <cstring>
# include <vector>
# include <boost/shared_array.hpp>

# include "blas_function.hpp"
# include "rand_function.hpp"

class Tensor {
public:
    Tensor(int n, int c, int h, int w, float std = 0) : data(new float[n * c * h * w]){
        N = n;
        C = c;
        H = h;
        W = w;
        if(std == 0) {
            memset(data.get(), 0, sizeof(float) * N * C * H * W);
        } else {
            gaussrand(0, std, data.get(), N*C*H*W);
        }
    }
    Tensor(const Tensor& T) {
        N = T.get_N();
        C = T.get_C();
        H = T.get_H();
        W = T.get_W();
        data = T.get_data();
    }
    boost::shared_array<float>  get_data() const {
        return data;
    }
    void set_data(boost::shared_array<float> data_input) {
        data = data_input;
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
    int get_size() const {
        return N * C * H * W;
    }
    void add_Tensor(Tensor& t, const float coeff_a, const float coeff_b) {
        vector_add(this->get_data().get(), t.get_data().get(), this->get_data().get(),
                   coeff_a, coeff_b, this->get_size());
    }
    bool operator==( Tensor const& t ) const {
        return std::equal(this->get_data().get(), this->get_data().get() + N * C * H * W,
           t.get_data().get());
    }
    bool operator!=( Tensor const& t ) const {
        return std::equal(this->get_data().get(), this->get_data().get() + N * C * H * W,
                          t.get_data().get());
    }
    Tensor& operator=(Tensor const& t) {
        if(this != &t) {
            this->N = t.N;
            this->C = t.C;
            this->H = t.H;
            this->W = t.W;
            this->data = t.get_data();
        }
        return *this;
    }
private:
    boost::shared_array<float> data;
    int N, C, H, W;
};
#endif //PYCONVNET_TENSOR_HPP
