# include <vector>

# include "convolution_layer.hpp"
# include "pooling_layer.hpp"
# include "relu_layer.hpp"
# include "Tensor.hpp"

typedef std::vector<Tensor> TensorVec;
//typedef std::vector<float> floatVec;

#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include <numpy/arrayobject.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

void init_from_numpy(Tensor& tensor_input, boost::python::object numpy_data) {
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(numpy_data.ptr());
    float* f_arr = static_cast<float*>(PyArray_DATA(data_arr));
    if (!(PyArray_FLAGS(data_arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        std::cout << "Input numpy data is not C contiguous" << std::endl;
        exit(-1);
    }
    int tensor_size = tensor_input.get_N() * tensor_input.get_C() *
            tensor_input.get_H() * tensor_input.get_W();
    boost::shared_array<float> data_ptr(new float[tensor_size]);
    std::copy(f_arr, f_arr + tensor_size, data_ptr.get());
    tensor_input.set_data(data_ptr);
}

void return_numpy(Tensor& tensor_input, boost::python::object numpy_data) {
    PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(numpy_data.ptr());
    float* f_arr = static_cast<float*>(PyArray_DATA(data_arr));
    if (!(PyArray_FLAGS(data_arr) & NPY_ARRAY_C_CONTIGUOUS)) {
        std::cout << "Input numpy data is not C contiguous" << std::endl;
        exit(-1);
    }
    boost::shared_array<float> data_ptr = tensor_input.get_data();
    int tensor_size = tensor_input.get_N() * tensor_input.get_C() *
            tensor_input.get_H() * tensor_input.get_W();
    std::copy(data_ptr.get(), data_ptr.get() + tensor_size, f_arr);
}

BOOST_PYTHON_MODULE(pylayer)
{
    using namespace boost::python;
    class_<std::vector<float> >("PyFloatVector")
            .def(vector_indexing_suite<std::vector<float>, true>());
    class_<Tensor>("PyTensor", init<int, int, int, int>())
            .def(init<Tensor&>())
            .def("init_from_numpy", &init_from_numpy)
            .def("return_numpy", &return_numpy)
            .add_property("N", &Tensor::get_N)
            .add_property("C", &Tensor::get_C)
            .add_property("H", &Tensor::get_H)
            .add_property("W", &Tensor::get_W);
    class_<TensorVec>("PyTensorVec")
            .def(vector_indexing_suite<std::vector<Tensor> >());
    class_<ConvolutionLayer>("PyConvolutionLayer",
                             init<int, int, int, int, int, int>())
            .def("forward", &ConvolutionLayer::forward)
            .def("backward", &ConvolutionLayer::backward);
    class_<PoolingLayer>("PyPoolingLayer",
                         init<int, int, int, int, int, int>())
            .def("forward", &PoolingLayer::forward)
            .def("backward", &PoolingLayer::backward);
    class_<ReLULayer>("PyReLULayer")
            .def("forward", &ReLULayer::forward)
            .def("forward", &ReLULayer::backward);
}
