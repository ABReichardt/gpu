// STB implementation lives in this TU by default; pass -DLANCZOS_NO_STB_IMPL
// when linking against a TU that already provides it (e.g. lanczos.cu).
#ifndef LANCZOS_NO_STB_IMPL
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "lanczos_common.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <execution>
#include <random>
#include <span>
#include <stdexcept>


template <int channels> std::vector<unsigned char> resample(const unsigned char* src,
                                    int src_w, int src_h,
                                    int dst_w, int dst_h, int a) {
    const Axis xs = build_axis(src_w, dst_w, a);
    const Axis ys = build_axis(src_h, dst_h, a);

    const int temp_row_step = dst_w * channels;
    std::vector<float> temp(src_h * temp_row_step);

    // Horizontal pass: uchar src -> float temp, one row per parallel task.
    {
        std::vector<int> rows(src_h);
        std::iota(rows.begin(), rows.end(), 0);
        std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](int y){
            const int src_row  = y * src_w * channels;
            const int temp_row = y * temp_row_step;

            for (int x_out = 0; x_out < dst_w; ++x_out){
                const int start  = axis_start(x_out, xs.inv_scale, xs.filter_radius, xs.taps, src_w);
                const int w_base = x_out * xs.taps;

                std::array<float, channels> acc{};
                for (int k = 0; k < xs.taps; ++k) {
                    const float wk = xs.weights[w_base + k];
                    const int p = src_row + (start + k) * channels;
                    for (int ch = 0; ch < channels; ++ch)
                        acc[ch] += wk * static_cast<float>(src[p + ch]);
                }
                for (int ch = 0; ch < channels; ++ch)
                    temp[temp_row + x_out * channels + ch] = acc[ch];
            }
        });
    }

    std::vector<unsigned char> dst(dst_w * dst_h * channels);
    // Vertical pass: float temp -> uchar dst. Mirrors the horizontal pass but
    // steps the taps across temp rows, and rounds+clamps on store.
    {
        std::vector<int> rows(dst_h);
        std::iota(rows.begin(), rows.end(), 0);
        std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&](int y_out){
            const int start   = axis_start(y_out, ys.inv_scale, ys.filter_radius, ys.taps, src_h);
            const int w_base  = y_out * ys.taps;
            const int dst_row = y_out * dst_w * channels;

            for (int x_out = 0; x_out < dst_w; ++x_out) {
                std::array<float, channels> acc{};
                for (int k = 0; k < ys.taps; ++k) {
                    const float wk = ys.weights[w_base + k];
                    const int p = (start + k) * temp_row_step + x_out * channels;
                    for (int ch = 0; ch < channels; ++ch)
                        acc[ch] += wk * temp[p + ch];
                }
                for (int ch = 0; ch < channels; ++ch) {
                    // Lanczos overshoots, so round then clamp into [0,255].
                    const int v = static_cast<int>(std::nearbyint(acc[ch]));
                    dst[dst_row + x_out * channels + ch] = static_cast<unsigned char>(std::clamp(v, 0, 255));
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

// I/O helpers (load_image, iends_with, write_image, StbImagePtr) live in
// lanczos_common.hpp so the GPU TU can share the same code.

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