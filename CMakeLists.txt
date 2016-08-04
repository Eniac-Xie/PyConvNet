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

include_directories( include/ )

set(SOURCE_FILES src/main.cpp include/im2col.hpp include/convolution_layer.hpp src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp)
set(TEST_FILES include/im2col.hpp include/convolution_layer.hpp src/convolution_layer.cpp include/Tensor.hpp include/blas_function.hpp src/blas_function.cpp test/test_conv_layer.cpp)

add_executable(PyConvNet ${SOURCE_FILES})
add_executable(TestConv ${TEST_FILES})

target_link_libraries (PyConvNet ${OpenBLAS_LIB})
target_link_libraries (TestConv ${OpenBLAS_LIB})
