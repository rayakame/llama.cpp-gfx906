#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv, bool l2 = false>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t, const ggml_cuda_ssm_conv_l2_args l2a) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    [[maybe_unused]] extern __shared__ float s_l2[];
    [[maybe_unused]] int l2r  = -1;
    [[maybe_unused]] int head = 0;
    if constexpr (l2) {
        const int c0 = bidy * split_d_inner;
#pragma unroll
        for (int r = 0; r < 2; r++) {
            if (r < l2a.n && c0 >= l2a.ch0[r] && c0 < l2a.ch0[r] + l2a.nheads[r] * (int) split_d_inner) {
                l2r  = r;
                head = (c0 - l2a.ch0[r]) / (int) split_d_inner;
            }
        }
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        const float y = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        y_block[i * stride_y + tid] = y;
        if constexpr (l2) {
            if (l2r >= 0) {
                s_l2[i * split_d_inner + tid] = y;
            }
        }
    }

    if constexpr (l2) {
        static_assert(split_d_inner == GGML_CUDA_SSM_CONV_L2_NCOLS, "a block must hold one head");
        if (l2r < 0) {
            return;
        }
        constexpr int lanes = split_d_inner / 4;
        float * s_scale = s_l2 + n_t * split_d_inner;
        const float eps       = l2r == 0 ? l2a.eps[0]    : l2a.eps[1];
        const float scale_out = l2r == 0 ? l2a.scale[0]  : l2a.scale[1];
        const int   nheads    = l2r == 0 ? l2a.nheads[0] : l2a.nheads[1];
        float *     dst_r     = l2r == 0 ? l2a.dst[0]    : l2a.dst[1];
        __syncthreads();

        // the sum of squares in the order of rms_norm_f32<32, 1, true>, so the result is bit-identical
        const int lane = tid % lanes;
        for (int t = tid / lanes; t < n_t; t += split_d_inner / lanes) {
            const float4 xs = ((const float4 *) (s_l2 + t * split_d_inner))[lane];
            float tmp = 0.0f;
            tmp += xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
            tmp = warp_reduce_sum<lanes>(tmp);
            if (lane == 0) {
                s_scale[t] = rsqrtf(tmp / l2a.ncols + eps);
            }
        }
        __syncthreads();

        float * l2_dst = dst_r + ((int64_t) bidx * n_t * nheads + head) * split_d_inner + tid;
        for (int64_t t = 0; t < n_t; t++) {
            l2_dst[t * nheads * split_d_inner] = (s_scale[t] * s_l2[t * split_d_inner + tid]) * scale_out;
        }
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// x is token-contiguous per channel and y channel-contiguous per token: the LDS tile keeps both sides coalesced.
template <bool apply_silu, int d_conv, int n_tok>
static __global__ void __launch_bounds__(256) ssm_conv_tile_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                                                const float * __restrict__ bias,
                                                                const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                                                float * __restrict__ dst, const int dst_nb1, const int dst_nb2,
                                                                const int n_t) {
    constexpr int n_ch       = 64;
    constexpr int n_threads  = 256;
    constexpr int n_cols     = d_conv - 1 + n_tok;
    // odd row stride: no bank conflicts when every lane reads its own channel row
    constexpr int row_stride = n_cols | 1;
    constexpr int n_iter     = (n_ch * n_cols + n_threads - 1) / n_threads;

    __shared__ float tile[(n_iter * n_threads + n_cols - 1) / n_cols * row_stride];

    ggml_cuda_pdl_lc();
    const int tid  = threadIdx.x;
    const int lane = tid % n_ch;
    const int t0   = blockIdx.z * n_tok;
    const int ch   = blockIdx.y * n_ch + lane;

    const int local_n_t  = min(n_tok, n_t - t0);
    const int local_cols = local_n_t + d_conv - 1;

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const float * x = (const float *) ((const char *) src0 + blockIdx.x * src0_nb2) + blockIdx.y * n_ch * stride_x + t0;
    float *       y = (float *) ((char *) dst + blockIdx.x * dst_nb2) + t0 * stride_y + ch;

    ggml_cuda_pdl_sync();
    // clamped, unguarded loads: all of them are in flight before the first LDS write
    float v[n_iter];
#pragma unroll
    for (int k = 0; k < n_iter; k++) {
        const unsigned e = k * n_threads + tid;
        const unsigned r = e / n_cols;
        const unsigned c = e - r * n_cols;
        v[k] = x[min(r, (unsigned) n_ch - 1) * stride_x + min(c, (unsigned) local_cols - 1)];
    }

    float w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        w[j] = src1[ch * stride_w + j];
    }
    const float b = bias != nullptr ? bias[ch] : 0.0f;

#pragma unroll
    for (int k = 0; k < n_iter; k++) {
        const unsigned e = k * n_threads + tid;
        const unsigned r = e / n_cols;
        const unsigned c = e - r * n_cols;
        tile[r * row_stride + c] = v[k];
    }
    __syncthreads();

    const float * row = tile + lane * row_stride;
#pragma unroll
    for (int k = 0; k < n_tok * n_ch / n_threads; k++) {
        const int t = k * (n_threads / n_ch) + tid / n_ch;
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; j++) {
            sumf += row[t + j] * w[j];
        }
        sumf += b;
        const float out = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        if (t < local_n_t) {
            y[t * stride_y] = out;
        }
    }
}

template <bool apply_silu, int d_conv, int n_tok>
static void ssm_conv_tile_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb1, const int src0_nb2,
                                   const int src1_nb1, float * dst, const int dst_nb1, const int dst_nb2, const int64_t nr,
                                   const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    const dim3 blocks(n_s, nr / 64, (n_t + n_tok - 1) / n_tok);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, 256, 0, stream);
    ggml_cuda_kernel_launch(ssm_conv_tile_f32<apply_silu, d_conv, n_tok>, launch_params, src0, src1, bias, src0_nb1, src0_nb2,
                            src1_nb1, dst, dst_nb1, dst_nb2, (int) n_t);
}

static bool ssm_conv_tile_enabled() {
    static const bool enabled = [] { const char * e = getenv("GGML_SSM_CONV_TILE"); return !e || atoi(e) != 0; }();
    return enabled;
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream, const ggml_cuda_ssm_conv_l2_args * l2 = nullptr) {
    const int threads = 128;

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if constexpr (apply_silu) {
            if (l2 != nullptr) {
                GGML_ASSERT(n_t <= GGML_CUDA_SSM_CONV_L2_MAX_T && l2->ncols == threads);
                const dim3   blocks(n_s, nr / threads, 1);
                const size_t smem = (n_t * threads + n_t) * sizeof(float);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, smem, stream);
                ggml_cuda_kernel_launch(ssm_conv_f32<true, threads, kNC, true>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t, *l2);
                return;
            }
        }
        if (ssm_conv_tile_enabled() && nr % 64 == 0 && n_t > 1) {
            if (n_t <= 4) {
                ssm_conv_tile_f32_cuda<apply_silu, kNC, 4>(src0, src1, bias, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb1, dst_nb2, nr, n_t, n_s, stream);
            } else if (n_t <= 8) {
                ssm_conv_tile_f32_cuda<apply_silu, kNC, 8>(src0, src1, bias, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb1, dst_nb2, nr, n_t, n_s, stream);
            } else if (n_t <= 16) {
                ssm_conv_tile_f32_cuda<apply_silu, kNC, 16>(src0, src1, bias, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb1, dst_nb2, nr, n_t, n_s, stream);
            } else if (n_t <= 32) {
                ssm_conv_tile_f32_cuda<apply_silu, kNC, 32>(src0, src1, bias, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb1, dst_nb2, nr, n_t, n_s, stream);
            } else {
                ssm_conv_tile_f32_cuda<apply_silu, kNC, 64>(src0, src1, bias, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb1, dst_nb2, nr, n_t, n_s, stream);
            }
            return;
        }
        GGML_ASSERT(nr % threads == 0);
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t,
                                                                        ggml_cuda_ssm_conv_l2_args{});
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 7:  launch_kernel(std::integral_constant<int, 7 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 7, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst,
                           const ggml_cuda_ssm_conv_l2_args * l2) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    GGML_ASSERT(l2 == nullptr || fuse_silu);
    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream, l2);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

template <int n_max>
static __global__ void ssm_conv_step_f32(const ggml_cuda_conv_step_args p) {
    constexpr int d_conv = 4;
    constexpr int n_old  = d_conv - 1;
    constexpr int n_hist = n_old + GGML_CUDA_CONV_STEP_MAX_WIN - 1;

    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= p.nr) {
        return;
    }

    // all reads come before the first write: a window may be the old state row, dst may be x
    const float * s = p.states + p.rows[0]*p.row_stride + c*n_old;
    float h[n_hist];
#pragma unroll
    for (int k = 0; k < n_hist; ++k) {
        h[k] = k < n_hist - n_old ? 0.0f : s[k - (n_hist - n_old)];
    }

    float xt[n_max];
#pragma unroll
    for (int t = 0; t < n_max; ++t) {
        xt[t] = t < p.n_t ? p.x[t*p.x_st + c*p.x_sc] : 0.0f;
    }

    float w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = p.w[c*p.w_sc + j];
    }

#pragma unroll
    for (int t = 0; t < n_max; ++t) {
        if (t >= p.n_t) {
            break;
        }
#pragma unroll
        for (int k = 0; k < n_hist - 1; ++k) {
            h[k] = h[k + 1];
        }
        h[n_hist - 1] = xt[t];

        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += h[n_hist - d_conv + j] * w[j];
        }
        // ssm_conv_f32 adds its zero bias, which turns -0 into +0
        sumf += 0.0f;
        p.dst[t*p.dst_st + c] = ggml_cuda_op_silu_single(sumf);
    }

#pragma unroll
    for (int r = 0; r < GGML_CUDA_CONV_STEP_MAX_WIN; ++r) {
        if (r < p.n_win) {
            float * d = p.win[r] + c*n_old;
#pragma unroll
            for (int j = 0; j < n_old; ++j) {
                d[j] = h[n_hist - n_old - r + j];
            }
        }
    }
}

void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_step_args & args) {
    GGML_ASSERT(args.n_t >= 1 && args.n_t <= GGML_CUDA_CONV_STEP_MAX_T);
    GGML_ASSERT(args.n_win >= 1 && args.n_win <= GGML_CUDA_CONV_STEP_MAX_WIN);

    const int    threads = 64;
    const dim3   blocks((args.nr + threads - 1) / threads);
    cudaStream_t stream  = ctx.stream();

    if (args.n_t == 1) {
        ssm_conv_step_f32<1><<<blocks, threads, 0, stream>>>(args);
    } else if (args.n_t <= 4) {
        ssm_conv_step_f32<4><<<blocks, threads, 0, stream>>>(args);
    } else if (args.n_t <= 8) {
        ssm_conv_step_f32<8><<<blocks, threads, 0, stream>>>(args);
    } else {
        ssm_conv_step_f32<GGML_CUDA_CONV_STEP_MAX_T><<<blocks, threads, 0, stream>>>(args);
    }
}
