// After mx-llama.cpp's q8_repack mmq_gemm_repacked (MIT), reading the weights straight from the Q8_0 planes.
#include "mmq-q8-0-planes-gemm.cuh"
#include "mmq.cuh"

#include <cstdlib>
#include <type_traits>

#define Q8P_BM  64 // weight rows per block
#define Q8P_BK  4  // 32-value blocks per K step
#define Q8P_CW  64 // columns per pass of the 64 lanes
#define Q8P_TN  2  // passes per block: 128 columns
#define Q8P_NRL 4  // waves, Q8P_BM/Q8P_NRL rows each

static_assert(Q8P_BK*QK8_0 == QK8_1_MMQ, "a K step is one q8_1 MMQ group");

struct q8p_xrow {
    uint4 q[Q8P_BK][2];
    uint4 pad;
};

// Moves the upper half of each 64-column pass by 16 rows so that the two halves of a wave hit other LDS banks.
static __device__ __forceinline__ int q8p_sx_swizzle(const int lr) {
    const int n  = lr >> 6;
    int       tx = lr & 63;
    tx ^= (tx >> 5) << 4;
    return (n << 6) | tx;
}

static __global__ void __launch_bounds__(Q8P_CW*Q8P_NRL, 2) mul_mat_q8_0_planes_gemm(
        const char * __restrict__ w, const block_q8_1_mmq * __restrict__ xq, float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1, const uint32_t nb01, const uint32_t n_tok, const uint32_t dst_s1,
        const bool vec_ok) {
#if defined(GGML_USE_HIP) && defined(GCN)
    constexpr int BN       = Q8P_CW*Q8P_TN;
    constexpr int NROW     = Q8P_BM/Q8P_NRL;
    constexpr int NTHREADS = Q8P_CW*Q8P_NRL;
    constexpr int W_EPT    = Q8P_BM*Q8P_BK/NTHREADS;
    constexpr int X_EPT    = BN*Q8P_BK/NTHREADS;
    static_assert(W_EPT*NTHREADS == Q8P_BM*Q8P_BK && X_EPT*NTHREADS == BN*Q8P_BK, "whole loads per thread");
    static_assert(NROW % 8 == 0, "eight f16 scales per LDS read");

    const int t  = threadIdx.x + threadIdx.y*blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const uint32_t row0  = blockIdx.x*Q8P_BM;
    const uint32_t tok0  = blockIdx.y*BN;
    const uint32_t n_sub = ne0/QK8_0;

    __shared__ uint4 sW_lo[Q8P_BM][Q8P_BK];
    __shared__ uint4 sW_hi[Q8P_BM][Q8P_BK];
    __shared__ __align__(16) uint16_t sWd[Q8P_BK][Q8P_BM];
    __shared__ q8p_xrow sXq[BN];
    __shared__ float    sXd[BN][Q8P_BK + 1];

    float acc[NROW][Q8P_TN] = {};

    uint4    pw_lo[W_EPT];
    uint4    pw_hi[W_EPT];
    uint16_t pw_d [W_EPT];
    uint4    px_lo[X_EPT];
    uint4    px_hi[X_EPT];
    float    px_d [X_EPT];

    // A group of 8 blocks in planes: their 256 quants, then their 8 f16 scales.
    auto load_w = [&](const uint32_t wrow, const uint32_t sb, uint4 & lo, uint4 & hi, uint16_t & d) {
        const char  * g = w + (size_t) wrow*nb01 + (size_t) (sb/8)*(8*sizeof(block_q8_0));
        const uint4 * q = (const uint4 *) (g + (sb % 8)*QK8_0);
        lo = q[0];
        hi = q[1];
        d  = ((const uint16_t *) (g + 8*QK8_0))[sb % 8];
    };

    auto prefetch = [&](const uint32_t sb0, const block_q8_1_mmq * x_grp, auto checked_tag) {
        constexpr bool checked = decltype(checked_tag)::value;
#pragma unroll
        for (int i = 0; i < W_EPT; ++i) {
            const int      e    = t + i*NTHREADS;
            const uint32_t wrow = row0 + e/Q8P_BK;
            if (!checked || wrow < ne1) {
                load_w(wrow, sb0 + e % Q8P_BK, pw_lo[i], pw_hi[i], pw_d[i]);
            } else {
                pw_lo[i] = make_uint4(0, 0, 0, 0);
                pw_hi[i] = make_uint4(0, 0, 0, 0);
                pw_d[i]  = 0;
            }
        }
#pragma unroll
        for (int i = 0; i < X_EPT; ++i) {
            const int      e   = t + i*NTHREADS;
            const uint32_t col = tok0 + e/Q8P_BK;
            const int      lk  = e % Q8P_BK;
            if (!checked || col < n_tok) {
                const block_q8_1_mmq & b = x_grp[col];
                const uint4          * q = (const uint4 *) (b.qs + lk*QK8_1);
                px_lo[i] = q[0];
                px_hi[i] = q[1];
                px_d[i]  = b.d4[lk];
            } else {
                px_lo[i] = make_uint4(0, 0, 0, 0);
                px_hi[i] = make_uint4(0, 0, 0, 0);
                px_d[i]  = 0.0f;
            }
        }
    };

    auto store = [&]() {
#pragma unroll
        for (int i = 0; i < W_EPT; ++i) {
            const int e  = t + i*NTHREADS;
            const int lr = e/Q8P_BK;
            const int lk = e % Q8P_BK;
            sW_lo[lr][lk] = pw_lo[i];
            sW_hi[lr][lk] = pw_hi[i];
            sWd[lk][lr]   = pw_d[i];
        }
#pragma unroll
        for (int i = 0; i < X_EPT; ++i) {
            const int e  = t + i*NTHREADS;
            const int lr = q8p_sx_swizzle(e/Q8P_BK);
            const int lk = e % Q8P_BK;
            sXq[lr].q[lk][0] = px_lo[i];
            sXq[lr].q[lk][1] = px_hi[i];
            sXd[lr][lk]      = px_d[i];
        }
    };

    const int sxc0 = q8p_sx_swizzle(tx);

    auto compute = [&]() {
        for (int kk = 0; kk < Q8P_BK; ++kk) {
            float dx[Q8P_TN];
            int   xq32[Q8P_TN][8];
#pragma unroll
            for (int n = 0; n < Q8P_TN; ++n) {
                const int   xc = sxc0 + n*Q8P_CW;
                const uint4 q0 = sXq[xc].q[kk][0];
                const uint4 q1 = sXq[xc].q[kk][1];
                dx[n] = sXd[xc][kk];
                xq32[n][0] = q0.x; xq32[n][1] = q0.y; xq32[n][2] = q0.z; xq32[n][3] = q0.w;
                xq32[n][4] = q1.x; xq32[n][5] = q1.y; xq32[n][6] = q1.z; xq32[n][7] = q1.w;
            }
            const int     r0  = ty*NROW;
            uint4         wlo = sW_lo[r0][kk];
            uint4         whi = sW_hi[r0][kk];
            const uint4 * dp  = (const uint4 *) &sWd[kk][r0];
#pragma unroll
            for (int g = 0; g < NROW; g += 8) {
                const uint4      dh4 = dp[g/8];
                const uint16_t * dh  = (const uint16_t *) &dh4;
#pragma unroll
                for (int rr = 0; rr < 8; ++rr) {
                    const int   r  = g + rr;
                    const uint4 wl = wlo;
                    const uint4 wh = whi;
                    if (r + 1 < NROW) {
                        wlo = sW_lo[r0 + r + 1][kk];
                        whi = sW_hi[r0 + r + 1][kk];
                    }
                    const float d = __half2float(*(const half *) &dh[rr]);
#pragma unroll
                    for (int n = 0; n < Q8P_TN; ++n) {
                        int idot = 0;
                        idot = ggml_cuda_dp4a((int) wl.x, xq32[n][0], idot);
                        idot = ggml_cuda_dp4a((int) wh.x, xq32[n][4], idot);
                        idot = ggml_cuda_dp4a((int) wl.y, xq32[n][1], idot);
                        idot = ggml_cuda_dp4a((int) wh.y, xq32[n][5], idot);
                        idot = ggml_cuda_dp4a((int) wl.z, xq32[n][2], idot);
                        idot = ggml_cuda_dp4a((int) wh.z, xq32[n][6], idot);
                        idot = ggml_cuda_dp4a((int) wl.w, xq32[n][3], idot);
                        idot = ggml_cuda_dp4a((int) wh.w, xq32[n][7], idot);
                        acc[r][n] += d*dx[n]*(float) idot;
                    }
                }
            }
        }
    };

    // One K step: store the prefetched step, prefetch the next one, then compute on the stored one.
    auto kloop = [&](auto checked_tag) {
        const block_q8_1_mmq * x_grp = xq;
        prefetch(0, x_grp, checked_tag);
        for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += Q8P_BK) {
            store();
            __syncthreads();
            if (sb0 + Q8P_BK < n_sub) {
                x_grp += n_tok;
                prefetch(sb0 + Q8P_BK, x_grp, checked_tag);
            }
            compute();
            __syncthreads();
        }
    };
    if (row0 + Q8P_BM <= ne1 && tok0 + BN <= n_tok) {
        kloop(std::false_type{});
    } else {
        kloop(std::true_type{});
    }

    const uint32_t rbase  = row0 + ty*NROW;
    const int      nvalid = min((int) ne1 - (int) rbase, NROW);
    if (nvalid <= 0) {
        return;
    }
#pragma unroll
    for (int n = 0; n < Q8P_TN; ++n) {
        const uint32_t tok = tok0 + tx + n*Q8P_CW;
        if (tok >= n_tok) {
            continue;
        }
        float * dst = y + (size_t) tok*dst_s1 + rbase;
#pragma unroll
        for (int g = 0; g < NROW; g += 4) {
            if (vec_ok && g + 4 <= nvalid) {
                *(float4 *) (dst + g) = make_float4(acc[g + 0][n], acc[g + 1][n], acc[g + 2][n], acc[g + 3][n]);
            } else {
#pragma unroll
                for (int r = g; r < g + 4; ++r) {
                    if (r < nvalid) {
                        dst[r] = acc[r][n];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(w, xq, y, ne0, ne1, nb01, n_tok, dst_s1, vec_ok);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(GCN)
}

bool ggml_cuda_q8_0_planes_gemm_use(const int64_t ne11) {
    static const bool    use      = [] { const char * e = getenv("GGML_CUDA_Q8_0_PLANES_GEMM");     return !e || atoi(e) != 0; }();
    // below 256 columns the planes MMQ tile is faster
    static const int64_t min_cols = [] { const char * e = getenv("GGML_CUDA_Q8_0_PLANES_GEMM_MIN"); return e ? atoll(e) : 256; }();
    return use && ne11 >= min_cols;
}

void ggml_cuda_q8_0_planes_gemm(const char * w, const void * xq, float * dst, const int64_t ne00, const int64_t ne01, const size_t nb01,
                                const int64_t ne11, const int64_t dst_s1, cudaStream_t stream) {
    GGML_ASSERT(ne00 % (8*QK8_0) == 0 && nb01 % 16 == 0 && (uintptr_t) w % 16 == 0 && (uintptr_t) xq % 16 == 0);
    const bool vec_ok = dst_s1 % 4 == 0 && (uintptr_t) dst % 16 == 0;
    const dim3 grid((ne01 + Q8P_BM - 1)/Q8P_BM, (ne11 + Q8P_CW*Q8P_TN - 1)/(Q8P_CW*Q8P_TN), 1);
    mul_mat_q8_0_planes_gemm<<<grid, dim3(Q8P_CW, Q8P_NRL, 1), 0, stream>>>(
        w, (const block_q8_1_mmq *) xq, dst, ne00, ne01, nb01, ne11, dst_s1, vec_ok);
}
