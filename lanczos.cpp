#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    const double px = std::numbers::pi * x;
    return std::sin(px) / px;
}

double lanczos(double x, int a) {
    if (x <= -a || x >= a) return 0.0;
    return sinc(x) * sinc(x / static_cast<double>(a));
}

// Put the weights into 1 big table
struct weights {
    int taps = 0;
    std::vector<int> starts;            // size of destination
    std::vector<float> weights;         // size of destination x taps
};

weights build_axis(int src_size, int dst_size, int a) {
    const double scale = static_cast<double> (dst_size) / src_size;
    const double inv_scale = 1.0 / scale;

    const double filter_radius = (scale < 1.0) ? a * inv_scale : static_cast<double>(a);
    const double kernel_step = (scale < 1.0) ? scale : 1.0;

    const int taps = std::min(
    2 * static_cast<int>(std::ceil(filter_radius)) + 1,
    src_size);

    weights wtable;
    wtable.taps = taps;
    wtable.starts.resize(dst_size);
    wtable.weights.assign(static_cast<std::size_t>(dst_size) * taps, 0.0f);
    
    for (int x = 0; x < dst_size; x++) {
        const double center = (x + 0.5) * inv_scale - 0.5;
        const int left = static_cast<int>(std::floor(center - filter_radius)) + 1;
        const int right = static_cast<int>(std::floor(center + filter_radius));
        const int start = std::clamp(left, 0, src_size - taps);

        wtable.starts[x] = start;

        float* row = &wtable.weights[static_cast<std::size_t>(x) * taps]; //current row of the weight table
        for (int k = 0; k < taps; ++k) {
            const int src_idx = start + k;
            
            if (src_idx >= left && src_idx <= right) {
                const double w = lanczos((src_idx - center) * kernel_step, a);
                row[k] = static_cast<float>(w);
            };
        }

        // Normalize
        const double sum = std::accumulate(row, row + taps, 0.0);
        if (sum != 0.0) {
            const float inv = static_cast<float>(1.0) / sum;
            for (int k = 0; k < taps; ++k) row[k] *= inv;
        }
    }
    return wtable;
}


template <int channels> std::vector<unsigned char> resample(const unsigned char* src, 
                                    int src_w, int src_h,
                                    int dst_w, int dst_h, int a) {
    const weights xs = build_axis(src_w, dst_w, a);
    const weights ys = build_axis(src_h, dst_h, a);

    // make a temporary vector for the intermediate image, and a value for the step size based on channels
    const std::size_t temp_row_step = static_cast<std::size_t>(dst_w) * channels;
    std::vector<float> temp(static_cast<std::size_t>(src_h) * temp_row_step);

    // parallel execution of rows
    {
        std::vector<int> rows(src_h);
        std::iota(rows.begin(), rows.end(),0);
        std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](int y){
            
            const unsigned char* src_row = src + static_cast<std::size_t>(y) * src_w * channels;
            
            // pointer for the start of the intermediate row
            float* temp_row = temp.data() + y * temp_row_step;

            for (int x_out = 0; x_out < dst_w; ++x_out){
                // get the source start index of the output pixel
                const int start = xs.starts[x_out];
                // get the pointer for the first weight of the output pixel, can then be stepped with w[0], w[1] ...
                const float* w = &xs.weights[static_cast<std::size_t>(x_out) * xs.taps];

                // per channel accumulator for the pixel
                std::array<float, channels> acc{};

                // iterate over the pixels that contribute from the source image
                for (int k = 0; k < xs.taps; ++k) {
                    // weight
                    const float wk = w[k];

                    // pointer for the pixel value at the first channel, can be stepped with p[0], p[1] ...
                    const unsigned char* p = src_row + static_cast<std::size_t>(start + k) * channels;
                    // accumulate the contributions for every cahnnel
                    for (int ch = 0; ch < channels; ++ch) {
                        acc[ch] += wk * static_cast<float>(p[ch]);
                    }
                }
                // write the accumulated pixel value into the intermediate vector using the temp_row pointer
                for (int ch = 0; ch < channels; ++ch) {
                    temp_row[x_out * channels + ch] = acc[ch];
                }

            }
        });
    }

    std::vector<unsigned char> dst(static_cast<std::size_t>(dst_w) * dst_h * channels);
    // Mostly same as horizontal pass
    {
        std::vector<int> rows(dst_h);
        std::iota(rows.begin(), rows.end(),0);
        std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](int y_out){
            const int start = ys.starts[y_out];
            const float* w = &ys.weights[static_cast<std::size_t>(y_out) * ys.taps];

            unsigned char* dst_row = dst.data() + static_cast<std::size_t>(y_out) * dst_w * channels;

            for (int x_out = 0; x_out < dst_w; ++x_out) {
                std::array<float, channels> acc{};

                for (int k = 0; k < ys.taps; ++k) {
                    const float wk = w[k];

                    // Iterate over the rows of temp
                    const float* p = temp.data()
                    + static_cast<std::size_t>(start + k) * temp_row_step  // rows
                    + static_cast<std::size_t>(x_out) * channels; // channels

                    for (int ch = 0; ch < channels; ++ch) {
                        acc[ch] += wk * p[ch];
                    }
                }
                for (int ch = 0; ch < channels; ++ch) {
                    // Round and clamp to RGB range
                    const int v = static_cast<int>(std::lround(acc[ch]));
                    dst_row[x_out * channels + ch] = static_cast<unsigned char>(std::clamp(v, 0, 255));
                }
            }
        });
    }
    return dst;
}



// 1,2,3 or 4 channel pictures
std::vector<unsigned char> resample(const unsigned char* src,
                                    int src_w, int src_h, int channels,
                                    int dst_w, int dst_h, int a) {
    switch (channels) {
        case 1: return resample<1>(src, src_w, src_h, dst_w, dst_h, a);
        case 2: return resample<2>(src, src_w, src_h, dst_w, dst_h, a);
        case 3: return resample<3>(src, src_w, src_h, dst_w, dst_h, a);
        case 4: return resample<4>(src, src_w, src_h, dst_w, dst_h, a);
        default:
            throw std::runtime_error("Unsupported channel count: "
                                     + std::to_string(channels));
    }
}

using StbImagePtr =
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)>;

StbImagePtr load_image(const std::string& path,
                       int& w, int& h, int& channels) {
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &channels, 0);
    return StbImagePtr(raw, stbi_image_free);
}

bool iends_with(std::string_view s, std::string_view suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a))
                               == std::tolower(static_cast<unsigned char>(b));
                      });
}

bool write_image(const std::string& path, int w, int h, int ch,
                 const unsigned char* data) {
    const int stride = w * ch;
    if (iends_with(path, ".png"))
        return stbi_write_png(path.c_str(), w, h, ch, data, stride) != 0;
    if (iends_with(path, ".bmp"))
        return stbi_write_bmp(path.c_str(), w, h, ch, data) != 0;
    if (iends_with(path, ".tga"))
        return stbi_write_tga(path.c_str(), w, h, ch, data) != 0;
    if (iends_with(path, ".jpg") || iends_with(path, ".jpeg"))
        return stbi_write_jpg(path.c_str(), w, h, ch, data, 95) != 0;
    std::cerr << "Unsupported output extension: " << path << "\n";
    return false;
}



#ifndef LANCZOS_NO_MAIN
int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <input> <output> <new_width> <new_height> [a]\n";
        return EXIT_FAILURE;
    }

    const std::string in_path  = argv[1];
    const std::string out_path = argv[2];
    const int dst_w = std::atoi(argv[3]);
    const int dst_h = std::atoi(argv[4]);
    const int a     = (argc == 6) ? std::atoi(argv[5]) : 3;

    if (dst_w <= 0 || dst_h <= 0 || a <= 0) {
        std::cerr << "new_width, new_height, and a must be positive.\n";
        return EXIT_FAILURE;
    }

    int src_w = 0, src_h = 0, channels = 0;
    auto pixels = load_image(in_path, src_w, src_h, channels);
    if (!pixels) {
        std::cerr << "Failed to load '" << in_path << "': "
                  << stbi_failure_reason() << "\n";
        return EXIT_FAILURE;
    }
    if (channels < 1 || channels > 4) {
        std::cerr << "Unsupported channel count: " << channels << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Loaded " << in_path << " (" << src_w << "x" << src_h
              << ", " << channels << " channels)\n"
              << "Resampling to " << dst_w << "x" << dst_h
              << " with Lanczos a=" << a << "\n";

    std::vector<unsigned char> out;
    try {
        out = resample(pixels.get(), src_w, src_h, channels,
                       dst_w, dst_h, a);
    } catch (const std::exception& e) {
        std::cerr << "Resample failed: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    if (!write_image(out_path, dst_w, dst_h, channels, out.data())) {
        std::cerr << "Failed to write '" << out_path << "'\n";
        return EXIT_FAILURE;
    }
    std::cout << "Wrote " << out_path << "\n";
    return EXIT_SUCCESS;
}
#endif