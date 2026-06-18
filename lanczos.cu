#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <type_traits>

#include "lanczos_common.hpp"

// Every CUDA call goes through this
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                             \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)




//   true  -> shared-memory tiled kernels (auto-falls back to global if the tile
//            won't fit in shared memory)
//   false -> always use the global-memory kernels
inline constexpr bool kUseSharedMemory = false;

// One place for the Lanczos dot-product. TIn = unsigned char (pass 1) or float (pass 2).
// in_u_stride = distance between consecutive taps along the resampled axis.
template <int channels, typename TIn>
__device__ inline void gather_taps(const TIn* in, int in_base, int in_u_stride,
                                   const float* __restrict__ weights, int w_base, int taps,
                                   float (&acc)[channels]) {
    if constexpr (channels == 4) {
        using Vec = std::conditional_t<std::is_same_v<TIn, float>, float4, uchar4>;
        const Vec* inV = reinterpret_cast<const Vec*>(in);
        const int b = in_base / 4, s = in_u_stride / 4;   // exact: channels==4
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < taps; ++k) {
            const float wk = weights[w_base + k];
            const Vec p = inV[b + k * s];
            a.x += wk * (float)p.x;  a.y += wk * (float)p.y;
            a.z += wk * (float)p.z;  a.w += wk * (float)p.w;
        }
        acc[0] = a.x; acc[1] = a.y; acc[2] = a.z; acc[3] = a.w;
    } else {
        #pragma unroll
        for (int ch = 0; ch < channels; ++ch) acc[ch] = 0.f;
        for (int k = 0; k < taps; ++k) {
            const float wk = weights[w_base + k];
            const int p = in_base + k * in_u_stride;
            #pragma unroll
            for (int ch = 0; ch < channels; ++ch)
                acc[ch] += wk * static_cast<float>(in[p + ch]);
        }
    }
}


template <int channels>
__global__ void resample_h(const unsigned char* __restrict__ src,
                           float* __restrict__ temp,
                           const float* __restrict__ xs_weights, int taps,
                           int src_w, int src_h, int dst_w,
                           double inv_scale, double filter_radius) {
    extern __shared__ unsigned char tile[];

    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y     = blockIdx.y * blockDim.y + threadIdx.y;


    const int x_first = blockIdx.x * blockDim.x;
    const int x_last  = min(x_first + (int)blockDim.x - 1, dst_w - 1);
    const int s_first = axis_start(x_first, inv_scale, filter_radius, taps, src_w);
    const int s_last  = axis_start(x_last,  inv_scale, filter_radius, taps, src_w) + taps;
    const int span    = s_last - s_first;

    // Cooperative load (every threadIdx.x helps), only for rows that exist.
    if (y < src_h) {
        const int src_row = y * src_w * channels;
        unsigned char* row_tile = tile + threadIdx.y * span * channels;
        for (int i = threadIdx.x; i < span * channels; i += blockDim.x)
            row_tile[i] = src[src_row + s_first * channels + i];
    }
    __syncthreads();                 

    if (x_out >= dst_w || y >= src_h) return;

    const int start   = axis_start(x_out, inv_scale, filter_radius, taps, src_w);
    const int in_base = (int)threadIdx.y * span * channels + (start - s_first) * channels;  // tile-local: row + tap offset
    float acc[channels];
    gather_taps<channels>(tile, in_base, channels, xs_weights, x_out * taps, taps, acc);

    const int out = y * dst_w * channels + x_out * channels;
    #pragma unroll
    for (int ch = 0; ch < channels; ++ch) temp[out + ch] = acc[ch];
}

template <int channels>
__global__ void resample_v(const float* __restrict__ temp, unsigned char* __restrict__ dst,
                           const float* __restrict__ ys_weights, int taps,
                           int dst_w, int dst_h, int src_h,
                           double inv_scale, double filter_radius) {
    extern __shared__ float ftile[];      // caches float `temp` rows

    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    const int y_first  = blockIdx.y * blockDim.y;
    const int y_last   = min(y_first + (int)blockDim.y - 1, dst_h - 1);
    const int r_first  = axis_start(y_first, inv_scale, filter_radius, taps, src_h);
    const int r_last   = axis_start(y_last,  inv_scale, filter_radius, taps, src_h) + taps;
    const int span_rows = r_last - r_first;          // [r_first,r_last) ⊆ [0,src_h]

    const int row_w = (int)blockDim.x * channels;    // tile row width
    const int row_step = dst_w * channels;           // temp row stride

    // Cooperative load: each thread fills its column for several rows.
    if (x_out < dst_w) {
        for (int r = threadIdx.y; r < span_rows; r += blockDim.y) {
            const int g_row = r_first + r;
            float* dstp = ftile + r * row_w + threadIdx.x * channels;
            const int srcp = g_row * row_step + x_out * channels;
            #pragma unroll
            for (int ch = 0; ch < channels; ++ch) dstp[ch] = temp[srcp + ch];
        }
    }
    __syncthreads();


    if (x_out >= dst_w || y_out >= dst_h) return;

    const int start   = axis_start(y_out, inv_scale, filter_radius, taps, src_h);
    const int in_base = (start - r_first) * row_w + (int)threadIdx.x * channels;  // tile-local column
    float acc[channels];
    gather_taps<channels>(ftile, in_base, row_w, ys_weights, y_out * taps, taps, acc);

    const int out = y_out * row_step + x_out * channels;
    #pragma unroll
    for (int ch = 0; ch < channels; ++ch) {
        const int v = __float2int_rn(acc[ch]);
        dst[out + ch] = (unsigned char)min(max(v, 0), 255);
    }
}

// Global-memory fallbacks: used when the per-block tile won't fit in shared
// memory (heavy downscaling -> large span/taps). Same math via gather_taps.
template <int channels>
__global__ void resample_h_global(const unsigned char* __restrict__ src,
                                  float* __restrict__ temp,
                                  const float* __restrict__ xs_weights, int taps,
                                  int src_w, int src_h, int dst_w,
                                  double inv_scale, double filter_radius) {
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y     = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_out >= dst_w || y >= src_h) return;

    const int start   = axis_start(x_out, inv_scale, filter_radius, taps, src_w);
    const int in_base = y * src_w * channels + start * channels;
    float acc[channels];
    gather_taps<channels>(src, in_base, channels, xs_weights, x_out * taps, taps, acc);

    const int out = y * dst_w * channels + x_out * channels;
    #pragma unroll
    for (int ch = 0; ch < channels; ++ch) temp[out + ch] = acc[ch];
}

template <int channels>
__global__ void resample_v_global(const float* __restrict__ temp,
                                  unsigned char* __restrict__ dst,
                                  const float* __restrict__ ys_weights, int taps,
                                  int dst_w, int dst_h, int src_h,
                                  double inv_scale, double filter_radius) {
    const int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_out >= dst_w || y_out >= dst_h) return;

    const int row_step = dst_w * channels;
    const int start    = axis_start(y_out, inv_scale, filter_radius, taps, src_h);
    const int in_base  = start * row_step + x_out * channels;
    float acc[channels];
    gather_taps<channels>(temp, in_base, row_step, ys_weights, y_out * taps, taps, acc);

    const int out = y_out * row_step + x_out * channels;
    #pragma unroll
    for (int ch = 0; ch < channels; ++ch) {
        const int v = __float2int_rn(acc[ch]);
        dst[out + ch] = (unsigned char)min(max(v, 0), 255);
    }
}


// Host launcher: build weight tables, move data to the device, run both passes,
// copy back, free.
template <int channels>
void run(const unsigned char* h_src, int src_w, int src_h,
         unsigned char* h_dst, int dst_w, int dst_h, int a) {
    const int src_bytes  = src_w * src_h * channels;
    const int temp_count = src_h * dst_w * channels;
    const int dst_bytes  = dst_w * dst_h * channels;

    const Axis xs = build_axis(src_w, dst_w, a);   
    const Axis ys = build_axis(src_h, dst_h, a);

    float* d_xs_weights = nullptr;
    float* d_ys_weights = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xs_weights, xs.weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ys_weights, ys.weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xs_weights, xs.weights.data(),
                        xs.weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys_weights, ys.weights.data(),
                        ys.weights.size() * sizeof(float), cudaMemcpyHostToDevice));



    // Device buffers
    unsigned char *d_src = nullptr, *d_dst = nullptr;
    float         *d_temp = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src,  src_bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_count * (int)sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst,  dst_bytes));

    CUDA_CHECK(cudaMemcpy(d_src, h_src, src_bytes, cudaMemcpyHostToDevice));

    const dim3 block(16, 16);

    // Pass 1: grid covers (dst_w columns) x (src_h source rows).
    const dim3 grid_h((dst_w + block.x - 1) / block.x,
                      (src_h + block.y - 1) / block.y);

    if constexpr (kUseSharedMemory) {
        int span_max = 0;
        for (int x0 = 0; x0 < dst_w; x0 += block.x) {
            const int xl = min(x0 + (int)block.x - 1, dst_w - 1);
            const int s0 = axis_start(x0, xs.inv_scale, xs.filter_radius, xs.taps, src_w);
            const int s1 = axis_start(xl, xs.inv_scale, xs.filter_radius, xs.taps, src_w) + xs.taps;
            span_max = std::max(span_max, s1 - s0);
        }
        const size_t shmem = (size_t)block.y * span_max * channels * sizeof(unsigned char);
        int max_shmem = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
        if (shmem <= (size_t)max_shmem)
            resample_h<channels><<<grid_h, block, shmem>>>(d_src, d_temp,
                                                d_xs_weights, xs.taps,
                                                src_w, src_h, dst_w,
                                                xs.inv_scale, xs.filter_radius);
        else  // tile too big for shared memory -> global fallback
            resample_h_global<channels><<<grid_h, block>>>(d_src, d_temp,
                                                d_xs_weights, xs.taps,
                                                src_w, src_h, dst_w,
                                                xs.inv_scale, xs.filter_radius);
    } else {
        resample_h_global<channels><<<grid_h, block>>>(d_src, d_temp,
                                            d_xs_weights, xs.taps,
                                            src_w, src_h, dst_w,
                                            xs.inv_scale, xs.filter_radius);
    }
    CUDA_CHECK(cudaGetLastError());


    // Pass 2: grid covers (dst_w columns) x (dst_h output rows).
    const dim3 grid_v((dst_w + block.x - 1) / block.x,
                      (dst_h + block.y - 1) / block.y);

    if constexpr (kUseSharedMemory) {
        int span_rows_max = 0;
        for (int y0 = 0; y0 < dst_h; y0 += block.y) {
            const int yl = min(y0 + (int)block.y - 1, dst_h - 1);
            const int r0 = axis_start(y0, ys.inv_scale, ys.filter_radius, ys.taps, src_h);
            const int r1 = axis_start(yl, ys.inv_scale, ys.filter_radius, ys.taps, src_h) + ys.taps;
            span_rows_max = std::max(span_rows_max, r1 - r0);
        }
        const size_t shmem_v = (size_t)span_rows_max * block.x * channels * sizeof(float);
        int max_shmem = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
        if (shmem_v <= (size_t)max_shmem)
            resample_v<channels><<<grid_v, block, shmem_v>>>(d_temp, d_dst, d_ys_weights,
                                                            ys.taps, dst_w, dst_h, src_h,
                                                            ys.inv_scale, ys.filter_radius);
        else  // tile too big for shared memory -> global fallback
            resample_v_global<channels><<<grid_v, block>>>(d_temp, d_dst, d_ys_weights,
                                                            ys.taps, dst_w, dst_h, src_h,
                                                            ys.inv_scale, ys.filter_radius);
    } else {
        resample_v_global<channels><<<grid_v, block>>>(d_temp, d_dst, d_ys_weights,
                                                        ys.taps, dst_w, dst_h, src_h,
                                                        ys.inv_scale, ys.filter_radius);
    }
    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));

    // Free everything 
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_dst));

    CUDA_CHECK(cudaFree(d_xs_weights));
    CUDA_CHECK(cudaFree(d_ys_weights));
}

// Channel dispatch, mirrors the CPU resample() overload.
void resample_cuda(const unsigned char* src, int src_w, int src_h, int channels,
                   unsigned char* dst, int dst_w, int dst_h, int a) {
    // std::printf("Launching CUDA resample with %d channels...\n", channels);
    switch (channels) {
        case 1: run<1>(src, src_w, src_h, dst, dst_w, dst_h, a); break;
        case 2: run<2>(src, src_w, src_h, dst, dst_w, dst_h, a); break;
        case 3: run<3>(src, src_w, src_h, dst, dst_w, dst_h, a); break;
        case 4: run<4>(src, src_w, src_h, dst, dst_w, dst_h, a); break;
        default:
            std::fprintf(stderr, "Unsupported channel count: %d\n", channels);
            std::exit(EXIT_FAILURE);
    }
}


// Correctness tests (known input -> known output, plus random CPU-vs-GPU).
static int max_abs_diff(const std::vector<unsigned char>& a,
                        const std::vector<unsigned char>& b) {
    int m = 0;
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
    return m;
}

// Reproducible random image (std::random, the pattern from the matmul slides).
static std::vector<unsigned char> random_image(int w, int h, int channels, unsigned seed) {
    std::vector<unsigned char> img(w * h * channels);
    std::mt19937 eng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& px : img) px = static_cast<unsigned char>(dist(eng));
    return img;
}

// Analytical 1: a constant image must resample to the same constant
// (verifies the weights are L1-normalized, edges included).
static bool test_constant(int channels) {
    const int sw = 64, sh = 48, dw = 200, dh = 150, a = 3;
    const unsigned char val = 137;
    std::vector<unsigned char> src(sw * sh * channels, val);
    std::vector<unsigned char> gpu(dw * dh * channels);
    resample_cuda(src.data(), sw, sh, channels, gpu.data(), dw, dh, a);
    int worst = 0;
    for (unsigned char c : gpu) worst = std::max(worst, std::abs((int)c - (int)val));
    std::printf("  [constant   ch=%d] %dx%d -> %dx%d : max|out-%d| = %d  %s\n",
                channels, sw, sh, dw, dh, (int)val, worst, worst <= 1 ? "PASS" : "FAIL");
    return worst <= 1;
}

// Analytical 2: identity resize (dst == src) must reproduce the input, because
// at scale 1 the Lanczos kernel collapses to a unit impulse.
static bool test_identity(int channels) {
    const int w = 96, h = 72, a = 3;
    const auto src = random_image(w, h, channels, 1234u);
    std::vector<unsigned char> gpu(w * h * channels);
    resample_cuda(src.data(), w, h, channels, gpu.data(), w, h, a);
    const int worst = max_abs_diff(src, gpu);
    std::printf("  [identity   ch=%d] %dx%d -> %dx%d : max|out-in| = %d  %s\n",
                channels, w, h, w, h, worst, worst <= 1 ? "PASS" : "FAIL");
    return worst <= 1;
}

// CPU-vs-GPU: a random input resampled both ways must agree within 1 LSB.
static bool test_cpu_vs_gpu(int channels, int sw, int sh, int dw, int dh, int a) {
    const auto src = random_image(sw, sh, channels, 9001u + channels);
    const auto cpu = resample(src.data(), sw, sh, channels, dw, dh, a);
    std::vector<unsigned char> gpu(dw * dh * channels);
    resample_cuda(src.data(), sw, sh, channels, gpu.data(), dw, dh, a);
    const int worst = max_abs_diff(cpu, gpu);
    std::printf("  [cpu-vs-gpu ch=%d] %dx%d -> %dx%d a=%d : max|cpu-gpu| = %d  %s\n",
                channels, sw, sh, dw, dh, a, worst, worst <= 1 ? "PASS" : "FAIL");
    return worst <= 1;
}

static int run_tests() {
    std::printf("Running correctness tests:\n");
    bool ok = true;
    for (int ch : {1, 3, 4}) {
        ok = test_constant(ch) && ok;
        ok = test_identity(ch) && ok;
    }
    ok = test_cpu_vs_gpu(3,  64,  64, 200, 150, 3) && ok;   // upscale, non-proportional
    ok = test_cpu_vs_gpu(3, 512, 512,  64,  64, 3) && ok;   // downscale
    ok = test_cpu_vs_gpu(4, 300, 200, 800, 600, 3) && ok;   // RGBA (vectorized path)
    std::printf("%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Benchmark: run each side `runs` times, report min/median/mean/max. The
// minimum is the robust figure
static int run_benchmark(const std::string& in_path, const std::string& out_path,
                         int dst_w, int dst_h, int a, int runs) {
    int src_w = 0, src_h = 0, channels = 0;
    auto pixels = load_image(in_path, src_w, src_h, channels);
    if (!pixels) {
        std::cerr << "Failed to load '" << in_path << "': " << stbi_failure_reason() << "\n";
        return EXIT_FAILURE;
    }
    if (channels < 1 || channels > 4) {
        std::cerr << "Unsupported channel count: " << channels << "\n";
        return EXIT_FAILURE;
    }
    std::cout << "Benchmark: " << src_w << "x" << src_h << "x" << channels
              << " -> " << dst_w << "x" << dst_h << ", a=" << a
              << ", runs=" << runs << "\n";

    using clock = std::chrono::steady_clock;
    using ms_d  = std::chrono::duration<double, std::milli>;

    std::vector<double> cpu_ms(runs), gpu_ms(runs);
    std::vector<unsigned char> cpu_out;
    std::vector<unsigned char> gpu_out(dst_w * dst_h * channels);

    for (int r = 0; r < runs; ++r) {
        const auto c0 = clock::now();
        cpu_out = resample(pixels.get(), src_w, src_h, channels, dst_w, dst_h, a);
        const auto c1 = clock::now();
        cpu_ms[r] = ms_d(c1 - c0).count();

        const auto g0 = clock::now();
        resample_cuda(pixels.get(), src_w, src_h, channels, gpu_out.data(), dst_w, dst_h, a);
        const auto g1 = clock::now();
        gpu_ms[r] = ms_d(g1 - g0).count();
    }

    // min / median / mean / max over the runs.
    auto stats = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double sum = 0; for (double x : v) sum += x;
        return std::array<double,4>{ v.front(), v[v.size()/2],
                                     sum / v.size(), v.back() };
    };
    const auto cs = stats(cpu_ms);
    const auto gs = stats(gpu_ms);

    std::printf("\n            %10s %10s %10s %10s\n", "min", "median", "mean", "max");
    std::printf("  CPU [ms]  %10.3f %10.3f %10.3f %10.3f\n", cs[0], cs[1], cs[2], cs[3]);
    std::printf("  GPU [ms]  %10.3f %10.3f %10.3f %10.3f\n", gs[0], gs[1], gs[2], gs[3]);
    std::printf("  speedup (min CPU / min GPU): %.2fx\n", cs[0] / gs[0]);

    if (!write_image(out_path, dst_w, dst_h, channels, gpu_out.data())) {
        std::cerr << "Failed to write '" << out_path << "'\n";
        return EXIT_FAILURE;
    }
    std::cout << "Wrote GPU output to " << out_path << "\n";
    return EXIT_SUCCESS;
}


// Main: --cpu / --gpu / --compare (resize), --bench (timing), --test (checks).
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <mode> ...\n"
              << "  --cpu     <in> <out> <w> <h> [a]\n"
              << "  --gpu     <in> <out> <w> <h> [a]\n"
              << "  --compare <in> <out> <w> <h> [a]\n"
              << "  --bench   <in> <out> <w> <h> <a> <runs>\n"
              << "  --test\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    const std::string mode = argv[1];

    // Self-contained correctness tests (no input file needed).
    if (mode == "--test")
        return run_tests();

    // Benchmark: --bench <in> <out> <w> <h> <a> <runs>
    if (mode == "--bench") {
        if (argc != 8) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        const int dst_w = std::atoi(argv[4]);
        const int dst_h = std::atoi(argv[5]);
        const int a     = std::atoi(argv[6]);
        const int runs  = std::atoi(argv[7]);
        if (dst_w <= 0 || dst_h <= 0 || a <= 0 || runs <= 0) {
            std::cerr << "new_width, new_height, a, and runs must be positive.\n";
            return EXIT_FAILURE;
        }
        return run_benchmark(argv[2], argv[3], dst_w, dst_h, a, runs);
    }

    // --cpu / --gpu / --compare share the same argument shape.
    if (argc < 6 || argc > 7 ||
        (mode != "--cpu" && mode != "--gpu" && mode != "--compare")) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string in_path  = argv[2];
    const std::string out_path = argv[3];
    const int dst_w = std::atoi(argv[4]);
    const int dst_h = std::atoi(argv[5]);
    const int a     = (argc == 7) ? std::atoi(argv[6]) : 3;

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

    // --cpu: mirror lanczos.cpp's original main exactly 
    if (mode == "--cpu") {
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

    // --gpu: GPU only, save output 
    if (mode == "--gpu") {
        std::vector<unsigned char> out(dst_w * dst_h * channels);
        resample_cuda(pixels.get(), src_w, src_h, channels,
                      out.data(), dst_w, dst_h, a);

        if (!write_image(out_path, dst_w, dst_h, channels, out.data())) {
            std::cerr << "Failed to write '" << out_path << "'\n";
            return EXIT_FAILURE;
        }
        std::cout << "Wrote " << out_path << "\n";
        return EXIT_SUCCESS;
    }

    // --compare: run both, compare, write GPU output
    // Each side is timed twice: the "cold" run includes one-shot setup (CUDA
    // context init on the GPU, TBB thread-pool spin-up on the CPU); the "warm"
    // run is the second call (compute + H<->D copies; it still allocates).
    using clock = std::chrono::steady_clock;
    using ms_d  = std::chrono::duration<double, std::milli>;

    // CPU cold + warm.
    const auto t_cpu_cold_start = clock::now();
    std::vector<unsigned char> cpu_out;
    try {
        cpu_out = resample(pixels.get(), src_w, src_h, channels,
                           dst_w, dst_h, a);
    } catch (const std::exception& e) {
        std::cerr << "CPU resample failed: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    const auto t_cpu_cold_end = clock::now();

    const auto t_cpu_warm_start = clock::now();
    cpu_out = resample(pixels.get(), src_w, src_h, channels,
                       dst_w, dst_h, a);
    const auto t_cpu_warm_end = clock::now();

    // GPU cold + warm.
    std::vector<unsigned char> gpu_out(dst_w * dst_h * channels);
    const auto t_gpu_cold_start = clock::now();
    resample_cuda(pixels.get(), src_w, src_h, channels,
                  gpu_out.data(), dst_w, dst_h, a);
    const auto t_gpu_cold_end = clock::now();

    const auto t_gpu_warm_start = clock::now();
    resample_cuda(pixels.get(), src_w, src_h, channels,
                  gpu_out.data(), dst_w, dst_h, a);
    const auto t_gpu_warm_end = clock::now();

    const double cpu_cold_ms = ms_d(t_cpu_cold_end - t_cpu_cold_start).count();
    const double cpu_warm_ms = ms_d(t_cpu_warm_end - t_cpu_warm_start).count();
    const double gpu_cold_ms = ms_d(t_gpu_cold_end - t_gpu_cold_start).count();
    const double gpu_warm_ms = ms_d(t_gpu_warm_end - t_gpu_warm_start).count();

    const int n          = static_cast<int>(cpu_out.size());
    int       max_diff   = 0;
    long long sum_sq     = 0;
    long long diff_count = 0;
    for (int i = 0; i < n; ++i) {
        const int d = std::abs(static_cast<int>(cpu_out[i]) -
                               static_cast<int>(gpu_out[i]));
        if (d > max_diff) max_diff = d;
        if (d > 0) ++diff_count;
        sum_sq += static_cast<long long>(d) * d;
    }
    const double rmse = std::sqrt(static_cast<double>(sum_sq) /
                                  static_cast<double>(n));

    std::printf("\nTiming:\n");
    std::printf("  CPU cold:  %8.2f ms\n", cpu_cold_ms);
    std::printf("  CPU warm:  %8.2f ms\n", cpu_warm_ms);
    std::printf("  GPU cold:  %8.2f ms  (incl. CUDA context init)\n", gpu_cold_ms);
    std::printf("  GPU warm:  %8.2f ms  (kernels + H<->D copies)\n", gpu_warm_ms);
    std::printf("  speedup (warm CPU / warm GPU): %.2fx\n",
                cpu_warm_ms / gpu_warm_ms);
    std::printf("\nCPU vs GPU comparison:\n");
    std::printf("  max abs diff:     %d / 255\n", max_diff);
    std::printf("  RMSE:             %.4f\n", rmse);
    std::printf("  pixels differing: %lld / %d (%.2f%%)\n",
                diff_count, n,
                100.0 * static_cast<double>(diff_count) /
                        static_cast<double>(n));
    std::printf("  verdict:          %s\n",
                max_diff <= 1 ? "PASS (within 1 LSB)"
                              : "FAIL (exceeds 1 LSB tolerance)");

    if (!write_image(out_path, dst_w, dst_h, channels, gpu_out.data())) {
        std::cerr << "Failed to write '" << out_path << "'\n";
        return EXIT_FAILURE;
    }
    std::cout << "Wrote GPU output to " << out_path << "\n";

    return max_diff <= 1 ? EXIT_SUCCESS : EXIT_FAILURE;
}
