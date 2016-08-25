# include <random>
# include "rand_function.hpp"

void gaussrand(float mean, float stddev, float* rand_array, int array_len) {
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(mean, stddev);
    for(int idx = 0; idx < array_len; ++idx) {
        rand_array[idx] = distribution(generator);
    }
}

