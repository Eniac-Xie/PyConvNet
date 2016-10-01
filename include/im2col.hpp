#ifndef PYCONVNET_IM2COL_HPP
#define PYCONVNET_IM2COL_HPP

#include <cstring>

void im2col(const float* data_input, const int height, const int width, const int channels,
            const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float* data_output) {
    int h_outout = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int w_output = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int output_width = h_outout * w_output;
    for (int base_h = 0; base_h < h_outout; ++base_h) {
        for (int base_w = 0; base_w < w_output; ++base_w) {
            for (int channels_idx = 0; channels_idx < channels; ++channels_idx) {
                for (int kernel_h_idx = 0; kernel_h_idx < kernel_h; ++kernel_h_idx) {
                    for (int kernel_w_idx = 0; kernel_w_idx < kernel_w; ++kernel_w_idx) {
                        int original_h = base_h * stride_h + kernel_h_idx - pad_h;
                        int original_w = base_w * stride_w + kernel_w_idx - pad_w;

                        int output_row = channels_idx * kernel_h * kernel_w +
                                         kernel_h_idx * kernel_w + kernel_w_idx;
                        int output_col = base_h * w_output + base_w;

                        if (original_h < 0 || original_h >= height ||
                                original_w < 0 || original_w >= width) {
                            data_output[output_row * output_width + output_col] = 0;
                        }
                        else {
                            data_output[output_row * output_width + output_col] =
                                    data_input[channels_idx * width * height +
                                            original_h * width + original_w];
                        }
                    }
                }
            }
        }
    }
}

void col2im(const float* data_col, const int height, const int width, const int channels,
            const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float* data_original) {
    memset(data_original, 0, sizeof(float) * height * width * channels);
    int data_col_height = channels * kernel_h * kernel_w;
    int data_col_width = ((height + 2 * pad_h - kernel_h) / stride_h + 1) *
            ((width + 2 * pad_w - kernel_w) / stride_w + 1);
    for (int row = 0; row < data_col_height; ++row) {
        for(int cols = 0; cols < data_col_width; ++cols) {
            int base_h = cols / ((width + 2 * pad_w - kernel_w) / stride_w + 1);
            int base_w = cols % ((width + 2 * pad_w - kernel_w) / stride_w + 1);
            int bias_channel = row / kernel_h / kernel_w;
            int bias_height = row % (kernel_h * kernel_w) / kernel_w;
            int bias_width = row % kernel_w;
            int original_h = base_h * stride_h + bias_height - pad_h;
            int original_w = base_w * stride_w + bias_width - pad_w;
            if (original_h < 0 || original_h >= height ||
                    original_w < 0 || original_w >= width)
                continue;
            data_original[bias_channel * height * width + original_h * width
                    + original_w] +=
                    data_col[row * data_col_width + cols];
        }
    }
}

#endif //PYCONVNET_IM2COL_HPP
