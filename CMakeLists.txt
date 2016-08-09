cmake_minimum_required(VERSION 3.3)
project(PyConvNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lopenblas")

set(Open_BLAS_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/include/openblas
        /usr/include/openblas-base
        /usr/local/include
        /usr/local/include/openblas
        /usr/local/include/openblas-base
        /opt/OpenBLAS/include
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/include
        )

set(Open_BLAS_LIB_SEARCH_PATHS
        /lib/
        /lib/openblas-base
        /lib64/
        /usr/lib
        /usr/lib/openblas-base
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/OpenBLAS/lib
        $ENV{OpenBLAS}cd
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
        )

find_path(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
find_library(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

find_package(PythonInterp)
find_package(PythonLibs)
find_package(Boost COMPONENTS python)
link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

set(NUMPY_INCLUDE_DIRS /home/xie/software/anaconda/lib/python2.7/site-packages/numpy/core/include)

include_directories( include/ ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

set(SOURCE_FILES src/main.cpp include/im2col.hpp include/convolution_layer.hpp src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp include/pooling_layer.hpp src/pooling_layer.cpp include/relu_layer.hpp src/relu_layer.cpp)
set(PYTHON_LAYER_FILES include/im2col.hpp include/convolution_layer.hpp src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp include/pooling_layer.hpp src/pooling_layer include/relu_layer.hpp src/relu_layer.cpp python/interfaces/layers.cpp)

set(TEST_CONV_FILES include/im2col.hpp include/convolution_layer.hpp src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp test/test_conv_layer.cpp)
set(TEST_POOLING_FILES include/Tensor.hpp include/pooling_layer.hpp src/pooling_layer.cpp test/test_pooling_layer.cpp)
set(TEST_RELU_FILES include/Tensor.hpp  include/relu_layer.hpp src/relu_layer.cpp test/test_relu_layer.cpp)

set(EXECUTABLE_OUTPUT_PATH bin/)

add_executable(PyConvNet ${SOURCE_FILES})
add_executable(TestConv ${TEST_CONV_FILES})
add_executable(TestPool ${TEST_POOLING_FILES})
add_executable(TestReLU ${TEST_RELU_FILES})

python_add_module(pylayer ${PYTHON_LAYER_FILES})

target_link_libraries (PyConvNet ${OpenBLAS_LIB})
target_link_libraries (TestConv ${OpenBLAS_LIB})
target_link_libraries (pylayer ${OpenBLAS_LIB})
