#pragma once

// Shared, host-side Lanczos math used by both the CPU reference (lanczos.cpp)
// and the GPU version (lanczos.cu). Header-only so a single nvcc translation
// unit can build both; functions are `inline` to stay ODR-safe when the CPU
// and GPU objects are linked together.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <memory>
#include <numbers>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

// stb headers are included here so both the CPU binary (lanczos.cpp) and the
// GPU binary (lanczos.cu) get the same I/O declarations. 
#include "stb_image.h"
#include "stb_image_write.h"

// host+device qualifier: expands to nothing for a plain C++ compiler, and to
// __host__ __device__ when compiled by nvcc.
#ifdef __CUDACC__
  #define LANCZOS_HD __host__ __device__
#else
  #define LANCZOS_HD
#endif

inline double sinc(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    const double px = std::numbers::pi * x;
    return std::sin(px) / px;
}

inline double lanczos(double x, int a) {
    if (x <= -a || x >= a) return 0.0;
    return sinc(x) * sinc(x / static_cast<double>(a));
}

struct Axis {
    int taps = 0;
    double inv_scale = 0.0;        // needed to recompute start on the fly
    double filter_radius = 0.0;
    std::vector<float> weights;         // size of destination x taps
};

// First source index feeding output sample x; host+device so the CPU and GPU
// compute a bit-identical start (weights are stored relative to it). The floor
// flips by one if (x+0.5)*inv_scale-0.5 rounds differently on host vs device, so
// both sides must disable FMA (-ffp-contract=off / nvcc --fmad=false).
LANCZOS_HD inline int axis_start(int x, double inv_scale, double filter_radius,
                                 int taps, int src_size) {
    const double center = (x + 0.5) * inv_scale - 0.5;
    const int    left   = static_cast<int>(std::floor(center - filter_radius)) + 1;
    const int    hi     = src_size - taps;
    return left < 0 ? 0 : (left > hi ? hi : left);
}


// Image I/O helpers (inline, header-only).
using StbImagePtr = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>;

inline StbImagePtr load_image(const std::string& path,
                              int& w, int& h, int& channels) {
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &channels, 0);
    return StbImagePtr(raw, stbi_image_free);
}

inline bool iends_with(std::string_view s, std::string_view suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
                      [](char a, char b) {
                          return std::tolower(static_cast<unsigned char>(a))
                               == std::tolower(static_cast<unsigned char>(b));
                      });
}

inline bool write_image(const std::string& path, int w, int h, int ch,
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

// Public resample APIs (CPU in lanczos.cpp, GPU in lanczos.cu).
std::vector<unsigned char> resample(const unsigned char* src,
                                    int src_w, int src_h, int channels,
                                    int dst_w, int dst_h, int a);

// GPU resample (defined in lanczos.cu).
void resample_cuda(const unsigned char* src, int src_w, int src_h, int channels,
                   unsigned char* dst, int dst_w, int dst_h, int a);

inline Axis build_axis(int src_size, int dst_size, int a) {
    const double scale = static_cast<double> (dst_size) / src_size;
    const double inv_scale = 1.0 / scale;

    const double filter_radius = (scale < 1.0) ? a * inv_scale : static_cast<double>(a);
    const double kernel_step = (scale < 1.0) ? scale : 1.0;

    const int taps = std::min(
    2 * static_cast<int>(std::ceil(filter_radius)) + 1,
    src_size);

    Axis wtable;
    wtable.taps = taps;
    wtable.inv_scale = inv_scale;
    wtable.filter_radius = filter_radius;
    wtable.weights.assign(dst_size * taps, 0.0f);

    for (int x = 0; x < dst_size; x++) {
        const double center = (x + 0.5) * inv_scale - 0.5;
        const int left = static_cast<int>(std::floor(center - filter_radius)) + 1;
        const int right = static_cast<int>(std::floor(center + filter_radius));
        const int start = axis_start(x, inv_scale, filter_radius, taps, src_size);

        const int base = x * taps; //current row of the weight table
        for (int k = 0; k < taps; ++k) {
            const int src_idx = start + k;

            if (src_idx >= left && src_idx <= right) {
                const double w = lanczos((src_idx - center) * kernel_step, a);
                wtable.weights[base + k] = static_cast<float>(w);
            };
        }

        // Normalize
        const double sum = std::accumulate(wtable.weights.begin() + base, wtable.weights.begin() + base + taps, 0.0);
        if (sum != 0.0) {
            const float inv = static_cast<float>(1.0) / sum;
            for (int k = 0; k < taps; ++k) wtable.weights[base + k] *= inv;
        }
    }
    return wtable;
}
