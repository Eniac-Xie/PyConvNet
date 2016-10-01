# include <iostream>
# include <fstream>
# include <cstring>
# include <boost/shared_array.hpp>
# include <sys/timeb.h>

# include "Layer.hpp"
# include "Net.hpp"
# include "mnist_demo.hpp"

float train_data[784*60000];
float train_label[60000];
float test_data[784*10000];
float test_label[10000];

static timeb TimingMilliSeconds;

void StartOfDuration() {
    ftime(&TimingMilliSeconds);
}

int EndOfDuration() {
    struct timeb now;
    ftime(&now);
    return int( (now.time-TimingMilliSeconds.time)*1000+(now.millitm-TimingMilliSeconds.millitm) );
}

int main() {
    // loading mnist data
    read_mnist("../../data/mnist/train-images-idx3-ubyte",
               "../../data/mnist/train-labels-idx1-ubyte",
               train_data, train_label);
    read_mnist("../../data/mnist/t10k-images-idx3-ubyte",
               "../../data/mnist/t10k-labels-idx1-ubyte",
               test_data, test_label);

    int batch_size = 100;
    std::shared_ptr<Net> lenet = init_lenet(batch_size);
    int epochs = 5;
    float lr_arr[5] = {0.0001, 0.0001, 0.00001, 0.00001, 0.00001};
    for (int epoch_idx = 0; epoch_idx < epochs; ++epoch_idx) {
        lenet->set_lr(lr_arr[epoch_idx]);
        for(int i = 0; i < (int)(60000 / batch_size); ++i) {
            float* batch_data_ptr = new float[batch_size * 28 * 28];
            float* batch_label_ptr = new float[batch_size];
            memcpy(batch_data_ptr, train_data + i * batch_size * 28 * 28,
                   sizeof(float) * batch_size * 28 * 28);
            memcpy(batch_label_ptr, train_label + i * batch_size, sizeof(float) * batch_size);
            boost::shared_array<float> data_batch(batch_data_ptr);
            boost::shared_array<float> label_batch(batch_label_ptr);
            Tensor train_data_tensor(batch_size, 1, 28, 28);
            Tensor train_label_tensor(batch_size, 1, 1, 1);
            train_data_tensor.set_data(data_batch);
            train_label_tensor.set_data(label_batch);
            std::cout << "epoch: " << epoch_idx + 1 << ", iter: " << i << ", ";
            StartOfDuration();
            lenet->train_batch(train_data_tensor, train_label_tensor);
            int msec = EndOfDuration();
            std::cout << msec << "ms elapse\n";
        }
    }

    // do testing
    float right_num = 0.f;
    for(int i = 0; i < (int)(10000 / batch_size); ++i) {
        float* batch_data_ptr = new float[batch_size * 28 * 28];
        float* batch_label_ptr = new float[batch_size];
        memcpy(batch_data_ptr, test_data + i * batch_size * 28 * 28,
               sizeof(float) * batch_size * 28 * 28);
        memcpy(batch_label_ptr, test_label + i * batch_size, sizeof(float) * batch_size);
        boost::shared_array<float> data_batch(batch_data_ptr);
        boost::shared_array<float> label_batch(batch_label_ptr);
        Tensor test_data_tensor(batch_size, 1, 28, 28);
        Tensor test_label_tensor(batch_size, 1, 1, 1);
        Tensor pred_label_tensor(batch_size, 1, 1, 1);
        test_data_tensor.set_data(data_batch);
        test_label_tensor.set_data(label_batch);
        std::cout << "test iter: " << i << "   ";
        StartOfDuration();
        lenet->test_batch(test_data_tensor, pred_label_tensor);
        int msec = EndOfDuration();
        float* label_ptr = test_label_tensor.get_data().get();
        float* pred_ptr = pred_label_tensor.get_data().get();
        for(int j = 0; j < batch_size; ++j) {
            if (label_ptr[j] == pred_ptr[j])
                ++right_num;
        }
        std::cout << msec << "ms elapse, batch acc: " << right_num / ((i+1) * batch_size) << std::endl;
    }
}
