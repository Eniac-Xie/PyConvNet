cmake_minimum_required(VERSION 3.3)
project(PyConvNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lopenblas -O2")

set(Open_BLAS_INCLUDE_SEARCH_PATHS $ENV{OpenBLAS_HOME})

set(Open_BLAS_LIB_SEARCH_PATHS $ENV{OpenBLAS_HOME})

find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
find_library(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

# find_package(PythonInterp)
# find_package(PythonLibs)
# find_package(Boost COMPONENTS python)
#link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
link_libraries(${Boost_LIBRARIES})

#set(NUMPY_INCLUDE_DIRS $ENV{ANACONDA_HOME}/lib/python2.7/site-packages/numpy/core/include)

#include_directories( include/ ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
include_directories( include/ ${Boost_INCLUDE_DIRS})

set(SOURCE_FILES include/Layer.hpp experiment/mnist/mnist_demo.cpp include/im2col.hpp include/convolution_layer.hpp
        src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp
        include/pooling_layer.hpp src/pooling_layer.cpp include/relu_layer.hpp src/relu_layer.cpp
        include/softmax_layer.hpp src/softmax_layer.cpp include/log_loss_layer.hpp src/log_loss_layer.cpp
        include/Net.hpp src/Net.cpp include/rand_function.hpp src/rand_function.cpp)
# set(PYTHON_LAYER_FILES include/Layer.hpp include/im2col.hpp include/convolution_layer.hpp
#         src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp
#         include/pooling_layer.hpp src/pooling_layer include/relu_layer.hpp src/relu_layer.cpp
#         python/interfaces/layers.cpp include/rand_function.hpp src/rand_function.cpp)

set(TEST_CONV_FILES include/Layer.hpp include/im2col.hpp include/convolution_layer.hpp
        src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp
        test/test_conv_layer.cpp include/rand_function.hpp src/rand_function.cpp)
set(TEST_POOLING_FILES include/Layer.hpp include/Tensor.hpp include/pooling_layer.hpp
        src/pooling_layer.cpp test/test_pooling_layer.cpp include/rand_function.hpp src/rand_function.cpp)
set(TEST_RELU_FILES include/Layer.hpp include/Tensor.hpp  include/relu_layer.hpp src/relu_layer.cpp
        test/test_relu_layer.cpp include/rand_function.hpp src/rand_function.cpp)
set(TEST_SOFTMAX_FILES include/Layer.hpp include/blas_function.hpp src/blas_function.cpp
        include/Tensor.hpp  include/softmax_layer.hpp src/softmax_layer.cpp
        include/log_loss_layer.hpp src/log_loss_layer.cpp test/test_softmax_log_loss_layer.cpp
        include/rand_function.hpp src/rand_function.cpp)

set(EXECUTABLE_OUTPUT_PATH bin/)

add_executable(MnistDemo ${SOURCE_FILES})
add_executable(TestConv ${TEST_CONV_FILES})
add_executable(TestPool ${TEST_POOLING_FILES})
add_executable(TestReLU ${TEST_RELU_FILES})
add_executable(TestSoftMaxLogLoss ${TEST_SOFTMAX_FILES})

# python_add_module(pylayer ${PYTHON_LAYER_FILES})

target_link_libraries (MnistDemo ${OpenBLAS_LIB})
target_link_libraries (TestConv ${OpenBLAS_LIB})
target_link_libraries(TestSoftMaxLogLoss ${OpenBLAS_LIB})
# target_link_libraries (pylayer ${OpenBLAS_LIB})
