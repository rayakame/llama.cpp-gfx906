#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type, bool aligned_b2 = false, bool iq_signs = false, bool grid_lds = false) {
#if defined(GGML_USE_HIP)
    if (iq_signs) {
        switch (type) {
            case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1_gcn;
            case GGML_TYPE_IQ3_XXS: return grid_lds ? vec_dot_iq3_xxs_q8_1_gcn<true> : vec_dot_iq3_xxs_q8_1_gcn<false>;
            case GGML_TYPE_IQ3_S:   return grid_lds ? vec_dot_iq3_s_q8_1_gcn<true> : vec_dot_iq3_s_q8_1_gcn<false>;
            default:                break;
        }
    }
#else
    GGML_UNUSED(iq_signs);
#endif // defined(GGML_USE_HIP)
    if (grid_lds) {
        switch (type) {
            case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1<true>;
            case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1<true>;
            default:                break;
        }
    }
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return aligned_b2 ? vec_dot_q4_0_q8_1_aligned : vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return aligned_b2 ? vec_dot_q8_0_q8_1_aligned : vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1<false>;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1<false>;
        default:                return nullptr;
    }
}

// GCN: Q4_K/Q5_K mat-vec lanes read a dwordx4 of weights (VDR 4, vecdotq.cuh); GGML_MMVQ_GCN_KVDR4=0 keeps VDR 2.
static constexpr __host__ __device__ bool mmvq_gcn_kvdr4_type(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
}

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda_gcn_kvdr4(ggml_type type) {
    return type == GGML_TYPE_Q4_K ? vec_dot_q4_K_q8_1_gcn : type == GGML_TYPE_Q5_K ? vec_dot_q5_K_q8_1_gcn : nullptr;
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

static constexpr __host__ __device__ bool mmvq_gcn_wide_type(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q8_0;
}

// GGML_MMVQ_GCN_WIDE=0 gives 9..16 columns to MMQ as before, GGML_MMVQ_GCN_WIDE_MAX=n takes the wide kernel up to n.
static int64_t mmvq_gcn_wide_max_batch(const ggml_type type) {
    static const int64_t max_batch = [] {
        const char * on  = getenv("GGML_MMVQ_GCN_WIDE");
        const char * max = getenv("GGML_MMVQ_GCN_WIDE_MAX");
        if (on && atoi(on) == 0) {
            return int64_t(0);
        }
        return max ? std::min<int64_t>(atoi(max), MMVQ_GCN_WIDE_MAX_BATCH) : int64_t(MMVQ_GCN_WIDE_MAX_BATCH);
    }();
    return mmvq_gcn_wide_type(type) ? max_batch : 0;
}

// GCN (MI50, Qwen3.8 27B shapes): the most columns for which the mat-vec beats MMQ's 8-column tile, 0 = not measured.
// With fewer rows MMQ has fewer tiles than the card has blocks and splits K (stream-k), so the mat-vec stays longer.
static constexpr int get_mmvq_max_batch_gcn(ggml_type type, bool wide) {
    switch (type) {
        case GGML_TYPE_Q8_0:    return 8;
        case GGML_TYPE_Q3_K:    return 8;
        case GGML_TYPE_Q4_K:    return wide ? 5 : 7;
        case GGML_TYPE_Q5_K:    return wide ? 5 : 8;
        case GGML_TYPE_Q6_K:    return 8;
        case GGML_TYPE_IQ3_XXS: return 8;
        case GGML_TYPE_IQ3_S:   return 8;
        case GGML_TYPE_IQ4_XS:  return 8;
        default:                return 0;
    }
}

// GGML_MMVQ_GCN_TABLE=0 keeps the mat-vec up to 8 columns for every type; GGML_MMVQ_GCN_MAX=n and
// GGML_MMVQ_GCN_MAX_<TYPE>=n (e.g. GGML_MMVQ_GCN_MAX_Q4_K) override the table for the types in it, at any row count;
// GGML_MMVQ_GCN_ROWS sets the row count from which the wide column of the table applies (8192).
static int get_mmvq_max_batch_gcn_env(const ggml_type type, const int64_t nrows_x) {
    const auto env = [](const std::string & name, const int def) {
        const char * s = getenv(name.c_str());
        return s ? atoi(s) : def;
    };
    static const int wide_rows = env("GGML_MMVQ_GCN_ROWS", 8192);
    static const std::array<std::array<int, 2>, GGML_TYPE_COUNT> max_batch = [&env] {
        const bool table = env("GGML_MMVQ_GCN_TABLE", 1) != 0;
        const int  all   = env("GGML_MMVQ_GCN_MAX", 0);
        std::array<std::array<int, 2>, GGML_TYPE_COUNT> r;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            for (int wide = 0; wide < 2; ++wide) {
                int n = MMVQ_MAX_BATCH_SIZE;
                if (get_mmvq_max_batch_gcn(ggml_type(t), wide) > 0) {
                    n = table ? get_mmvq_max_batch_gcn(ggml_type(t), wide) : n;
                    n = all > 0 ? all : n;
                    std::string name = std::string("GGML_MMVQ_GCN_MAX_") + ggml_type_name(ggml_type(t));
                    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
                    n = env(name, n);
                }
                r[t][wide] = std::clamp(n, 1, MMVQ_MAX_BATCH_SIZE);
            }
        }
        return r;
    }();
    return max_batch[type][nrows_x >= wide_rows];
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11, int64_t nrows_x) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
#if defined(GGML_USE_HIP)
    if (GGML_CUDA_CC_IS_GCN(cc) && ne11 > MMVQ_MAX_BATCH_SIZE) {
        return ne11 <= mmvq_gcn_wide_max_batch(type);
    }
#endif // defined(GGML_USE_HIP)
    if (GGML_CUDA_CC_IS_GCN(cc)) {
        return ne11 <= get_mmvq_max_batch_gcn_env(type, nrows_x);
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

// Rows per block and warps for the GCN register-cached Q4_0/Q4_1 path: 2..4 columns (the MTP verify pass) and 5..8.
#ifndef GCN_MMVQ_Q4_RPB
#define GCN_MMVQ_Q4_RPB 4
#endif
#ifndef GCN_MMVQ_Q4_NW
#define GCN_MMVQ_Q4_NW 1
#endif
#ifndef GCN_MMVQ_Q4_RPB8
#define GCN_MMVQ_Q4_RPB8 4
#endif
#ifndef GCN_MMVQ_Q4_NW8
#define GCN_MMVQ_Q4_NW8 1
#endif
// 5..8 columns read their weights faster with half a block per thread, whatever VDR the type uses elsewhere.
#ifndef GCN_MMVQ_Q4_VDR8
#define GCN_MMVQ_Q4_VDR8 2
#endif

static constexpr __host__ __device__ bool mmvq_gcn_q4_multicol(ggml_type type, int ncols_dst, int table_id) {
    return table_id == MMVQ_PARAMETERS_GCN && (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1) && ncols_dst >= 2 && ncols_dst <= 8;
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        if (mmvq_gcn_q4_multicol(type, ncols_dst, table_id)) {
            return ncols_dst <= 4 ? GCN_MMVQ_Q4_NW : GCN_MMVQ_Q4_NW8;
        }
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (mmvq_gcn_q4_multicol(type, ncols_dst, table_id)) {
        return ncols_dst <= 4 ? GCN_MMVQ_Q4_RPB : GCN_MMVQ_Q4_RPB8;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return table_id == MMVQ_PARAMETERS_GENERIC && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool aligned_b2 = false, bool planes = false, bool iq_signs = false, bool grid_lds = false, bool kvdr4 = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    // planes at 1..2 columns: half a block per lane, one 16-byte load
    constexpr int vdr = kvdr4 ? VDR_Q4_K_Q5_K_Q8_1_MMVQ_GCN : planes && ncols_dst <= 2 ? 4 :
        mmvq_gcn_q4_multicol(type, ncols_dst, table_id) && ncols_dst > 4 ? GCN_MMVQ_Q4_VDR8 : get_vdr_mmvq(type);
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = kvdr4 ? get_vec_dot_q_cuda_gcn_kvdr4(type) : get_vec_dot_q_cuda(type, aligned_b2, iq_signs, grid_lds);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    if constexpr (grid_lds) {
        ggml_cuda_iq_grid_lds_fill<type, nwarps*warp_size>(tid);
        __syncthreads();
    }

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    // planes: rows start on a group and kbx steps by whole groups, so a lane keeps its block in the group
    static_assert(!planes || blocks_per_iter % 8 == 0, "planes need whole groups per K step");
    [[maybe_unused]] const int planes_j = (tid / (qi/vdr)) & 7;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#if defined(GGML_USE_HIP) && defined(GCN)
        // Each column's activations are otherwise re-read through L1 for every row; on Vega that read is the
        // bottleneck of the 4-column MTP verify pass (after JeanP00l/mi50-qwen3.8-fast, ported to Q4_1 and the gate).
        if constexpr (mmvq_gcn_q4_multicol(type, ncols_dst, table_id)) {
            constexpr bool is_q4_0 = type == GGML_TYPE_Q4_0;
            int   v[rows_per_cuda_block][vdr];
            int   vg[has_fusion ? rows_per_cuda_block : 1][vdr];
            half2 dmx[rows_per_cuda_block];
            half2 dmg[has_fusion ? rows_per_cuda_block : 1];
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const int ib = kbx_offset + i*stride_row_x + kbx;
                if constexpr (is_q4_0) {
                    const block_q4_0 * bx = (const block_q4_0 *) vx + ib;
#pragma unroll
                    for (int l = 0; l < vdr; ++l) {
                        v[i][l] = get_int_b2(bx->qs, kqs + l);
                    }
                    dmx[i] = make_half2(bx->d, bx->d);
                } else {
                    const block_q4_1 * bx = (const block_q4_1 *) vx + ib;
#pragma unroll
                    for (int l = 0; l < vdr; ++l) {
                        v[i][l] = get_int_b4(bx->qs, kqs + l);
                    }
                    dmx[i] = bx->dm;
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        if constexpr (is_q4_0) {
                            const block_q4_0 * bg = (const block_q4_0 *) vgate + ib;
#pragma unroll
                            for (int l = 0; l < vdr; ++l) {
                                vg[i][l] = get_int_b2(bg->qs, kqs + l);
                            }
                            dmg[i] = make_half2(bg->d, bg->d);
                        } else {
                            const block_q4_1 * bg = (const block_q4_1 *) vgate + ib;
#pragma unroll
                            for (int l = 0; l < vdr; ++l) {
                                vg[i][l] = get_int_b4(bg->qs, kqs + l);
                            }
                            dmg[i] = bg->dm;
                        }
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const block_q8_1 * by = &y[j*stride_col_y + kby];
                int u[2*vdr];
#pragma unroll
                for (int l = 0; l < vdr; ++l) {
                    u[2*l+0] = get_int_b4(by->qs, kqs + l);
                    u[2*l+1] = get_int_b4(by->qs, kqs + l + qi);
                }
                const half2 ds = by->ds;
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if constexpr (is_q4_0) {
                        tmp[j][i] += vec_dot_q4_0_q8_1_impl<vdr>(v[i], u, __low2float(dmx[i]), ds);
                    } else {
                        tmp[j][i] += vec_dot_q4_1_q8_1_impl<vdr>(v[i], u, dmx[i], ds);
                    }
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            if constexpr (is_q4_0) {
                                tmp_gate[j][i] += vec_dot_q4_0_q8_1_impl<vdr>(vg[i], u, __low2float(dmg[i]), ds);
                            } else {
                                tmp_gate[j][i] += vec_dot_q4_1_q8_1_impl<vdr>(vg[i], u, dmg[i], ds);
                            }
                        }
                    }
                }
            }
            continue;
        }
#endif // defined(GGML_USE_HIP) && defined(GCN)

        if constexpr (planes) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const char * xb = (const char *) ((const block_q8_0 *) vx + kbx_offset + i*stride_row_x + kbx);
                int v[vdr];
                ggml_cuda_memcpy_1<sizeof(v)>(v, (const int *) __builtin_assume_aligned(xb - 2*planes_j, 16) + kqs);
                const half d = *(const half *) (xb + 8*QK8_0 - 32*planes_j);
                [[maybe_unused]] int   vg[vdr];
                [[maybe_unused]] half  dg = d;
                if constexpr (has_fusion) {
                    if (use_gate) {
                        const char * gb = (const char *) ((const block_q8_0 *) vgate + kbx_offset + i*stride_row_x + kbx);
                        ggml_cuda_memcpy_1<sizeof(vg)>(vg, (const int *) __builtin_assume_aligned(gb - 2*planes_j, 16) + kqs);
                        dg = *(const half *) (gb + 8*QK8_0 - 32*planes_j);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const block_q8_1 * by = &y[j*stride_col_y + kby];
                    int u[vdr];
#pragma unroll
                    for (int l = 0; l < vdr; ++l) {
                        u[l] = get_int_b4(by->qs, kqs + l);
                    }
                    tmp[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(v, u, d, __low2half(by->ds));
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q8_0_q8_1_impl<float, vdr>(vg, u, dg, __low2half(by->ds));
                        }
                    }
                }
            }
            continue;
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

#if defined(GGML_USE_HIP)
// GCN at 2..8 columns: Vega's 16 KB L1 does not keep the activations beside the streamed weights, so their L2 reads
// (1.8x the weight bytes for Q4_1 at 4 columns, 4 rows per wave) were the limit; ngroups row groups share them.
static constexpr __host__ __device__ int mmvq_gcn_mc_log2(const int x) {
    return x <= 1 ? 0 : 1 + mmvq_gcn_mc_log2(x/2);
}

static constexpr __host__ __device__ int mmvq_gcn_mc_nscatter(const int nvals, const int nlanes) {
    return mmvq_gcn_mc_log2(nvals) < mmvq_gcn_mc_log2(nlanes) ? mmvq_gcn_mc_log2(nvals) : mmvq_gcn_mc_log2(nlanes);
}

// Sums a[] over aligned groups of nlanes lanes with ~nvals shuffles; afterwards a[k] holds value
// mmvq_gcn_mc_first(lane) + k for k < nvals >> nscatter, and runs of keep values stay on one lane.
template <int nvals, int nlanes, int keep = 1, int s = 0>
static __device__ __forceinline__ void mmvq_gcn_mc_reduce(float * a, const int lane) {
    if constexpr ((nlanes >> s) > 1) {
        constexpr int off = nlanes >> (s + 1);
        constexpr int h   = nvals  >> (s + 1);
        if constexpr (h >= keep) {
            const bool up = lane & off;
#pragma unroll
            for (int k = 0; k < h; ++k) {
                const float send = up ? a[k]     : a[k + h];
                const float mine = up ? a[k + h] : a[k];
                a[k] = mine + __shfl_xor(send, off, 64);
            }
        } else {
#pragma unroll
            for (int k = 0; k < keep; ++k) {
                a[k] += __shfl_xor(a[k], off, 64);
            }
        }
        mmvq_gcn_mc_reduce<nvals, nlanes, keep, s + 1>(a, lane);
    }
}

template <int nvals, int nlanes, int keep>
static __device__ __forceinline__ int mmvq_gcn_mc_first(const int lane) {
    int first = 0;
#pragma unroll
    for (int s = 0; s < mmvq_gcn_mc_nscatter(nvals/keep, nlanes); ++s) {
        first += lane & (nlanes >> (s + 1)) ? nvals >> (s + 1) : 0;
    }
    return first;
}

static constexpr __host__ __device__ int mmvq_gcn_mc_ncols_p2(const int ncols_dst) {
    return ncols_dst == 1 ? 1 : (ncols_dst <= 2 ? 2 : (ncols_dst <= 4 ? 4 : 8));
}

// Types that mul_mat_vec_q_gcn_mc reads one whole block per lane; the others go through mul_mat_vec_q_gcn_grp.
static constexpr __host__ __device__ bool mmvq_gcn_mc_whole(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q8_0;
}

static constexpr __host__ __device__ bool mmvq_gcn_mc_q4(const ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1;
}

// Q4_K holds 2 ints per lane and row: 4 rows share each activation read up to 4 columns, past that the compiler
// no longer keeps all column loads in flight with 4 rows.
static constexpr __host__ __device__ int mmvq_gcn_mc_rows(const ggml_type type, const int ncols_dst, const bool gate = false) {
    return mmvq_gcn_mc_q4(type) ? (gate ? 2 : 4) : type == GGML_TYPE_Q8_0 ? 4 : (type == GGML_TYPE_Q4_K && ncols_dst <= 4 ? 4 : 2);
}

// a[(j*rows + i)*nper + m]: partial sums over this lane's K blocks of column j, row row_r + i, of the matrix (m = 0)
// and, fused, of the gate (m = 1). Rows below row_g belong to the previous group, which moved back from past the end.
template <int ncols_dst, int rows, int nlanes, int nwarps, bool gate>
static __device__ __forceinline__ void mmvq_gcn_mc_store(
        float * a, float * dst, const float * x_bias, const float * gate_bias, const ggml_glu_op glu_op,
        const float glu_limit, const uint32_t stride_col_dst, const int row_g, const int row_r) {
    constexpr int nper     = gate ? 2 : 1;
    constexpr int nvals    = mmvq_gcn_mc_ncols_p2(ncols_dst)*rows*nper;
    constexpr int nscatter = mmvq_gcn_mc_nscatter(nvals/nper, nlanes);
    constexpr int nout     = nvals >> nscatter;

    const int lane = threadIdx.x;
    const int lg   = lane % nlanes;

    mmvq_gcn_mc_reduce<nvals, nlanes, nper>(a, lg);

    if constexpr (nwarps > 1) {
        __shared__ float red[nwarps - 1][nout][64];
        if (threadIdx.y > 0) {
#pragma unroll
            for (int k = 0; k < nout; ++k) {
                red[threadIdx.y - 1][k][lane] = a[k];
            }
        }
        __syncthreads();
        if (threadIdx.y > 0) {
            return;
        }
#pragma unroll
        for (int w = 0; w < nwarps - 1; ++w) {
#pragma unroll
            for (int k = 0; k < nout; ++k) {
                a[k] += red[w][k][lane];
            }
        }
    }

    if ((lg & ((nlanes >> nscatter) - 1)) != 0) {
        return;
    }
    const int first = mmvq_gcn_mc_first<nvals, nlanes, nper>(lg);
#pragma unroll
    for (int k = 0; k < nout; k += nper) {
        const int o = (first + k) / nper;
        const int j = o / rows;
        const int i = o % rows;
        if (j < ncols_dst && row_r + i >= row_g) {
            const int id = j*stride_col_dst + row_r + i;
            float result = a[k];
            if (x_bias) {
                result += x_bias[id];
            }
            if constexpr (gate) {
                float gate_value = a[k + 1];
                if (gate_bias) {
                    gate_value += gate_bias[id];
                }
                switch (glu_op) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            } else {
                GGML_UNUSED_VARS(gate_bias, glu_op, glu_limit);
            }
            dst[id] = result;
        }
    }
}

template <ggml_type type, int ncols_dst, int ngroups, int nwarps, int waves_per_eu, bool gate, bool aligned_b2, bool planes = false>
__launch_bounds__(nwarps*64, waves_per_eu)
static __global__ void mul_mat_vec_q_gcn_mc(
        const void * __restrict__ vx, const void * __restrict__ vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
#if defined(GCN)
    static_assert(mmvq_gcn_mc_whole(type), "Q4_0/Q4_1/Q8_0 only");
    constexpr bool is_q4_0  = type == GGML_TYPE_Q4_0;
    constexpr bool is_q8_0  = type == GGML_TYPE_Q8_0;
    static_assert(!(is_q8_0 && gate), "gate fusion is Q4 only");
    static_assert(!planes || is_q8_0, "planes are Q8_0 only");
    constexpr int  nper     = gate ? 2 : 1;
    constexpr int  rows     = mmvq_gcn_mc_rows(type, ncols_dst, gate);
    constexpr int  qi       = is_q8_0 ? QI8_0 : 4;
    constexpr int  nu       = is_q8_0 ? qi : 2*qi;
    constexpr int  nlanes   = 64/ngroups;
    constexpr int  nvals    = mmvq_gcn_mc_ncols_p2(ncols_dst)*rows*nper;
    constexpr int  bs       = is_q4_0 ? sizeof(block_q4_0) : is_q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_1);

    const int lane  = threadIdx.x;
    const int lg    = lane % nlanes;
    const int rowb  = blockIdx.x*ngroups*rows;
    const int row_g = rowb + (lane/nlanes)*rows;
    // a group past the end moves back to end at the last row and writes only its own rows; nrows_x >= rows
    const int row_r = row_g + rows <= int(nrows_x) ? row_g : int(nrows_x) - rows;
    const int rowb_r = rowb < int(nrows_x) - rows ? rowb : int(nrows_x) - rows;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const size_t xoff = (size_t(sample_x*stride_sample_x + channel_x*stride_channel_x) + size_t(rowb_r)*stride_row_x)*bs;
    const char * x0 = (const char *) vx + xoff;
    [[maybe_unused]] const char * g0 = gate ? (const char *) fusion.gate + xoff : nullptr;
    const block_q8_1 * y = (const block_q8_1 *) vy + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const uint32_t row_bytes = stride_row_x*bs;
    const uint32_t offx0     = uint32_t(row_r - rowb_r)*row_bytes;

    float a[nvals] = {0.0f};

    const int blocks_per_row = ncols_x/ggml_cuda_type_traits<type>::qk;
    for (int kb = threadIdx.y*nlanes + lg; kb < blocks_per_row; kb += nwarps*nlanes) {
        // weight row r: row r/nper of the matrix (r%nper == 0) or of the gate
        int   v[rows*nper][qi];
        half2 dm[rows*nper];
#pragma unroll
        for (int r = 0; r < rows*nper; ++r) {
            const int i = r/nper;
            const char * bxc = (r % nper ? g0 : x0) + (offx0 + uint32_t(kb)*bs + i*row_bytes);
            if constexpr (planes) {
                const char * qs = (const char *) __builtin_assume_aligned(bxc - 2*(kb & 7), 16);
                ggml_cuda_memcpy_1<16>(v[r] + 0, qs +  0);
                ggml_cuda_memcpy_1<16>(v[r] + 4, qs + 16);
                const half d = *(const half *) (bxc + 8*QK8_0 - 32*(kb & 7));
                dm[r] = make_half2(d, d);
            } else if constexpr (is_q4_0 && aligned_b2) {
                half d;
                get_block_b2_aligned18(bxc, v[r], d);
                dm[r] = make_half2(d, d);
            } else if constexpr (is_q4_0 || is_q8_0) {
                using block_t = std::conditional_t<is_q4_0, block_q4_0, block_q8_0>;
                const block_t * bx = (const block_t *) bxc;
#pragma unroll
                for (int l = 0; l < qi; ++l) {
                    v[r][l] = get_int_b2(bx->qs, l);
                }
                dm[r] = make_half2(bx->d, bx->d);
            } else {
                const block_q4_1 * bx = (const block_q4_1 *) bxc;
#pragma unroll
                for (int l = 0; l < qi; ++l) {
                    v[r][l] = get_int_b4(bx->qs, l);
                }
                dm[r] = bx->dm;
            }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * by = (const block_q8_1 *) ((const char *) (y + j*stride_col_y) + uint32_t(kb)*uint32_t(sizeof(block_q8_1)));
            int u[nu];
#pragma unroll
            for (int l = 0; l < qi; ++l) {
                if constexpr (is_q8_0) {
                    u[l] = get_int_b4(by->qs, l);
                } else {
                    u[2*l+0] = get_int_b4(by->qs, l);
                    u[2*l+1] = get_int_b4(by->qs, l + qi);
                }
            }
            const half2 ds = by->ds;
#pragma unroll
            for (int r = 0; r < rows*nper; ++r) {
                if constexpr (is_q4_0) {
                    a[j*rows*nper + r] += vec_dot_q4_0_q8_1_impl<qi>(v[r], u, __low2float(dm[r]), ds);
                } else if constexpr (is_q8_0) {
                    a[j*rows*nper + r] += vec_dot_q8_0_q8_1_impl<float, qi>(v[r], u, __low2float(dm[r]), __low2float(ds));
                } else {
                    a[j*rows*nper + r] += vec_dot_q4_1_q8_1_impl<qi>(v[r], u, dm[r], ds);
                }
            }
        }
    }

    const uint32_t dst_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    mmvq_gcn_mc_store<ncols_dst, rows, nlanes, nwarps, gate>(a, dst + dst_offset,
        fusion.x_bias ? (const float *) fusion.x_bias + dst_offset : nullptr,
        gate && fusion.gate_bias ? (const float *) fusion.gate_bias + dst_offset : nullptr,
        fusion.glu_op, fusion.glu_limit, stride_col_dst, row_g, row_r);
#else
    GGML_UNUSED_VARS(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
        stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst);
    NO_DEVICE_CODE;
#endif // defined(GCN)
}

using mmvq_gcn_i32x4 = __attribute__((ext_vector_type(4))) int;

// buf[l]: the 16-byte chunk src[l] of the q8_1 tile that starts at superblock kb; a superblock past the end of the row
// loads the first one of the tile again, nobody reads it
template <int qk, int nld, int nsb, int cpc>
static __device__ __forceinline__ void mmvq_gcn_grp_tile_load(mmvq_gcn_i32x4 * buf, const char * y, const uint32_t * src,
        const int * sbc, const int kb, const int blocks_per_row) {
    const uint32_t o = kb*uint32_t((qk/QK8_1)*sizeof(block_q8_1));
#pragma unroll
    for (int l = 0; l < nld; ++l) {
        const bool ok = nsb == 1 || kb + sbc[l] < blocks_per_row;
        buf[l] = *(const mmvq_gcn_i32x4 *) (y + (o + (ok ? src[l] : src[l] - sbc[l]*(cpc/nsb)*16)));
    }
}

template <ggml_type type, int ncols_dst, int ngroups, int nwarps, int waves_per_eu, bool lds, bool iq_signs = false>
__launch_bounds__(nwarps*64, waves_per_eu)
static __global__ void mul_mat_vec_q_gcn_grp(
        const void * __restrict__ vx, const void * __restrict__ vy, const ggml_cuda_mm_fusion_args_device fusion,
        float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
#if defined(GCN)
    constexpr bool gate  = false;
    constexpr int qk     = ggml_cuda_type_traits<type>::qk;
    constexpr int qi     = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr    = get_vdr_mmvq(type);
    constexpr int rows   = mmvq_gcn_mc_rows(type, ncols_dst);
    constexpr int nlanes = 64/ngroups;
    constexpr int lpb    = qi/vdr;
    static_assert(nlanes % lpb == 0, "a group must cover whole blocks");
    constexpr int nvals  = mmvq_gcn_mc_ncols_p2(ncols_dst)*rows;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type, false, iq_signs);

    const int lane  = threadIdx.x;
    const int lg    = lane % nlanes;
    const int rowb  = blockIdx.x*ngroups*rows;
    const int row_g = rowb + (lane/nlanes)*rows;
    const int row_r = row_g + rows <= int(nrows_x) ? row_g : int(nrows_x) - rows;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const block_q8_1 * y = (const block_q8_1 *) vy + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row_r*stride_row_x;

    float a[nvals] = {0.0f};

    const int blocks_per_row = ncols_x/qk;
    const int lk  = threadIdx.y*nlanes + lg;
    const int kqs = vdr*(lk % lpb);
    if constexpr (lds) {
        // The row groups of a wave read the same activations: the wave copies them once per K step into LDS.
        constexpr int nsb    = nlanes/lpb;
        constexpr int nby    = nsb*(qk/QK8_1);
        constexpr int cpc    = nby*int(sizeof(block_q8_1))/16;
        constexpr int nchunk = ncols_dst*cpc;
        constexpr int nld    = (nchunk + 63)/64;
        constexpr int kstep  = nwarps*nsb;
        static_assert(nby*sizeof(block_q8_1) % 16 == 0, "the tile of a column must be whole 16-byte chunks");
        __shared__ mmvq_gcn_i32x4 tile[nwarps][nld*64];
        const block_q8_1 * ty = (const block_q8_1 *) tile[threadIdx.y] + (lg/lpb)*(qk/QK8_1);

        uint32_t src[nld];
        int      sbc[nld];
#pragma unroll
        for (int l = 0; l < nld; ++l) {
            const int c = min(l*64 + lane, nchunk - 1);
            src[l] = (c/cpc)*stride_col_y*uint32_t(sizeof(block_q8_1)) + (c % cpc)*16;
            sbc[l] = (c % cpc)/(cpc/nsb);
        }
        for (int kb0 = threadIdx.y*nsb; kb0 < blocks_per_row; kb0 += kstep) {
            mmvq_gcn_i32x4 buf[nld];
            mmvq_gcn_grp_tile_load<qk, nld, nsb, cpc>(buf, (const char *) y, src, sbc, kb0, blocks_per_row);
#pragma unroll
            for (int l = 0; l < nld; ++l) {
                tile[threadIdx.y][l*64 + lane] = buf[l];
            }
            // no barrier: a wave runs its LDS accesses in order, and one would keep the weight loads behind the copy
            const int kbx = kb0 + lg/lpb;
            if (nsb == 1 || kbx < blocks_per_row) {
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                    for (int i = 0; i < rows; ++i) {
                        a[j*rows + i] += vec_dot_q_cuda(vx, ty + j*nby, kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    } else {
        for (int kbx = lk/lpb; kbx < blocks_per_row; kbx += nwarps*nlanes/lpb) {
            const int kby = kbx*(qk/QK8_1);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    a[j*rows + i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    const uint32_t dst_offset = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
    mmvq_gcn_mc_store<ncols_dst, rows, nlanes, nwarps, gate>(a, dst + dst_offset,
        fusion.x_bias ? (const float *) fusion.x_bias + dst_offset : nullptr,
        gate && fusion.gate_bias ? (const float *) fusion.gate_bias + dst_offset : nullptr,
        fusion.glu_op, fusion.glu_limit, stride_col_dst, row_g, row_r);
#else
    GGML_UNUSED_VARS(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, channel_ratio,
        stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst);
    NO_DEVICE_CODE;
#endif // defined(GCN)
}

// 9..16 columns: one row per lane and the waves of a block split K, so the activations are the same for the whole
// wave and come in through scalar loads; MMQ stages the weights through LDS for only 16 columns of work there.
// y has the MMQ layout, so the columns of a 128-value chunk lie 144 bytes apart. Columns ncols_dst..ncols-1 read past
// the used ones (the buffer has room) and are not written, so that the loop has no branches.
template <ggml_type type, int ncols, int nwarps, int nb>
__launch_bounds__(nwarps*64, 4)
static __global__ void mul_mat_vec_q_gcn_wide(
        const void * __restrict__ vx, const block_q8_1_mmq * __restrict__ vy, float * __restrict__ dst,
        const int ncols_dst, const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t stride_row_x,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
#if defined(GCN)
    static_assert(mmvq_gcn_wide_type(type), "Q4_0/Q4_1/Q8_0 only");
    static_assert(4 % nb == 0, "a step must stay in one 128-value chunk");
    constexpr bool is_q4_0 = type == GGML_TYPE_Q4_0;
    constexpr bool is_q8_0 = type == GGML_TYPE_Q8_0;
    constexpr int  nv      = 8;
    using block_t = std::conditional_t<is_q4_0, block_q4_0, std::conditional_t<is_q8_0, block_q8_0, block_q4_1>>;

    const int lane  = threadIdx.x;
    const int w     = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int row   = blockIdx.x*64 + lane;
    const int row_r = row < int(nrows_x) ? row : int(nrows_x) - 1;

    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);

    const block_t        * x = (const block_t *) vx + size_t(sample_x*stride_sample_x + channel_x*stride_channel_x) + size_t(row_r)*stride_row_x;
    const block_q8_1_mmq * y = vy + sample_dst*stride_sample_y + channel_dst*stride_channel_y;

    float a[ncols] = {0.0f};

    const int blocks_per_row = ncols_x/ggml_cuda_type_traits<type>::qk;
    for (int kb0 = w*nb; kb0 < blocks_per_row; kb0 += nwarps*nb) {
        // Q4: v[b][l] and v[b][l + 4] are the low and high nibbles of int l, which meet y ints l and l + 4
        int    v[nb][nv];
        float2 dm[nb];
#pragma unroll
        for (int b = 0; b < nb; ++b) {
            const block_t * bx = x + (kb0 + b < blocks_per_row ? kb0 + b : kb0);
            if constexpr (is_q8_0) {
#pragma unroll
                for (int l = 0; l < nv; ++l) {
                    v[b][l] = get_int_b2(bx->qs, l);
                }
                dm[b] = make_float2(__half2float(bx->d), 0.0f);
            } else {
#pragma unroll
                for (int l = 0; l < nv/2; ++l) {
                    const int q = is_q4_0 ? get_int_b2(bx->qs, l) : get_int_b4(bx->qs, l);
                    v[b][l]        = (q >> 0) & 0x0F0F0F0F;
                    v[b][l + nv/2] = (q >> 4) & 0x0F0F0F0F;
                }
                if constexpr (is_q4_0) {
                    const float d = __half2float(bx->d);
                    dm[b] = make_float2(d, -8.0f*d);
                } else {
                    dm[b] = __half22float2(bx->dm);
                }
            }
        }
        const block_q8_1_mmq * yc = y + (kb0/4)*ncols_dst;
        const int              bc = kb0 % 4;
#pragma unroll
        for (int b = 0; b < nb; ++b) {
            if (nb > 1 && kb0 + b >= blocks_per_row) {
                break;
            }
            // the next column's activations are loaded before this column's dot, to hide the scalar load latency
            int   u[2][nv];
            float ud[2][2];
            const auto load_y = [&](const int j, int * uj, float * udj) {
                const int * yq = (const int *) yc[j].qs + (bc + b)*nv;
#pragma unroll
                for (int l = 0; l < nv; ++l) {
                    uj[l] = yq[l];
                }
                if constexpr (is_q8_0) {
                    udj[0] = yc[j].d4[bc + b];
                } else {
                    const float2 ds = __half22float2(yc[j].ds4[bc + b]);
                    udj[0] = ds.x;
                    udj[1] = ds.y;
                }
            };
            load_y(0, u[0], ud[0]);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                if constexpr (ncols < 16) {
                    // ROCm 10.1 hoists all columns' loads at 9 and 12 columns and spills SGPRs (not at 16)
                    __builtin_amdgcn_sched_barrier(0);
                }
                if (j + 1 < ncols) {
                    load_y(j + 1, u[(j + 1) % 2], ud[(j + 1) % 2]);
                }
                const int * uj  = u[j % 2];
                const float * udj = ud[j % 2];
                int sumi = 0;
#pragma unroll
                for (int l = 0; l < nv; ++l) {
                    sumi = ggml_cuda_dp4a(v[b][l], uj[l], sumi);
                }
                // Q4_0: dm.y = -8*d, Q4_1: dm.y = m; y's sum term goes into the sum first
                if constexpr (is_q8_0) {
                    a[j] = fmaf(float(sumi), dm[b].x*udj[0], a[j]);
                } else {
                    a[j] = fmaf(float(sumi), dm[b].x*udj[0], fmaf(dm[b].y, udj[1], a[j]));
                }
            }
        }
    }

    if constexpr (nwarps > 1) {
        __shared__ float red[nwarps/2][ncols][64];
#pragma unroll
        for (int s = nwarps/2; s >= 1; s /= 2) {
            if (w >= s && w < 2*s) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    red[w - s][j][lane] = a[j];
                }
            }
            __syncthreads();
            if (w < s) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    a[j] += red[w][j][lane];
                }
            }
            if (s > 1) {
                __syncthreads();
            }
        }
    }

    if (w != 0 || row >= int(nrows_x)) {
        return;
    }
    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst;
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (j < ncols_dst) {
            dst[j*stride_col_dst + row] = a[j];
        }
    }
#else
    GGML_UNUSED_VARS(vx, vy, dst, ncols_dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, channel_ratio,
        stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y,
        stride_sample_dst);
    NO_DEVICE_CODE;
#endif // defined(GCN)
}
#endif // defined(GGML_USE_HIP)

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

#if defined(GGML_USE_HIP)
// The types that reach the row-group kernel with their own vec_dot, each with its own switch.
static constexpr const char * mmvq_gcn_mc_type_env(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:    return "GGML_MMVQ_GCN_MC_Q2K";
        case GGML_TYPE_Q3_K:    return "GGML_MMVQ_GCN_MC_Q3K";
        case GGML_TYPE_IQ2_S:   return "GGML_MMVQ_GCN_MC_IQ2S";
        case GGML_TYPE_IQ3_XXS: return "GGML_MMVQ_GCN_MC_IQ3XXS";
        case GGML_TYPE_IQ3_S:   return "GGML_MMVQ_GCN_MC_IQ3S";
        case GGML_TYPE_IQ4_NL:  return "GGML_MMVQ_GCN_MC_IQ4NL";
        case GGML_TYPE_IQ4_XS:  return "GGML_MMVQ_GCN_MC_IQ4XS";
        default:                return nullptr;
    }
}

// Q4 at 2 and 5 columns fits 84 VGPRs (3 waves per SIMD) either way, 6..8 spill there; 3..4 hoist all loads at 2
// waves. The K-quants stay below 84 VGPRs up to 5 columns anyway. One column: Q6_K fits 40 VGPRs, Q4 48.
// Q8_0 takes the Q4 shapes but spills at 3 columns and 3 waves. Q4_K at 5 loads the columns one after another at 3 waves.
// IQ, Q2_K and Q3_K: up to 128 VGPRs they issue their loads together; where they need more, the compiler waits for
// each load in turn at 60-90 VGPRs, and those 2-4 waves per SIMD beat 1 wave of 130-220 VGPRs.
static constexpr int mmvq_gcn_mc_wpe(const ggml_type type, const int ncols_dst) {
    if (ncols_dst == 1) {
        return mmvq_gcn_mc_whole(type) ? 5 : 6;
    }
    if (mmvq_gcn_mc_type_env(type) != nullptr) {
        return 2;
    }
    return ncols_dst == 2 || (ncols_dst == 5 && type != GGML_TYPE_Q4_K) ||
        (!mmvq_gcn_mc_whole(type) && mmvq_gcn_mc_rows(type, ncols_dst) == 2 && ncols_dst <= 4) ? 3 : 2;
}

// Q4 at 3..4 columns and 3 waves per SIMD loads the columns one after another; 0 = no second variant.
static constexpr int mmvq_gcn_mc_wpe_alt(const ggml_type type, const int ncols_dst) {
    return mmvq_gcn_mc_whole(type) && (ncols_dst == 4 || (ncols_dst == 3 && mmvq_gcn_mc_q4(type))) ? 3 : 0;
}

static int mmvq_gcn_mc_env(const char * name, const int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

struct mmvq_gcn_mc_args {
    const void * vx;
    const void * vy;
    ggml_cuda_mm_fusion_args_device fusion;
    float * dst;
    int ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst;
    uint3 channel_ratio;
    int stride_channel_x, stride_channel_y, stride_channel_dst;
    uint3 sample_ratio;
    int stride_sample_x, stride_sample_y, stride_sample_dst;
    int nchannels_dst, nsamples_dst;
    bool planes;
    bool iq_signs;
};

template <ggml_type type, int c_ncols_dst, int ngroups, int nwarps, int waves_per_eu, bool gate, bool aligned_b2, bool planes = false>
static void mul_mat_vec_q_gcn_mc_launch(const mmvq_gcn_mc_args & p, cudaStream_t stream) {
    constexpr int rows = mmvq_gcn_mc_rows(type, c_ncols_dst, gate);
    const int64_t nblocks = (p.nrows_x + rows*ngroups - 1) / (rows*ngroups);
    const dim3 block_nums(nblocks, p.nchannels_dst, p.nsamples_dst);
    const dim3 block_dims(64, nwarps, 1);
    if constexpr (mmvq_gcn_mc_whole(type)) {
        mul_mat_vec_q_gcn_mc<type, c_ncols_dst, ngroups, nwarps, waves_per_eu, gate, aligned_b2, planes><<<block_nums, block_dims, 0, stream>>>(
            p.vx, p.vy, p.fusion, p.dst, p.ncols_x, p.nrows_x, p.stride_row_x, p.stride_col_y, p.stride_col_dst,
            p.channel_ratio, p.stride_channel_x, p.stride_channel_y, p.stride_channel_dst, p.sample_ratio,
            p.stride_sample_x, p.stride_sample_y, p.stride_sample_dst);
    } else {
        static_assert(!gate, "gate fusion is Q4 only");
        static const bool lds = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_LDS", 1) != 0;
        auto kernel = mul_mat_vec_q_gcn_grp<type, c_ncols_dst, ngroups, nwarps, waves_per_eu, false>;
        // measured for Q5_K/Q6_K only
        if constexpr (c_ncols_dst > 1 && (type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K)) {
            if (lds) {
                kernel = mul_mat_vec_q_gcn_grp<type, c_ncols_dst, ngroups, nwarps, waves_per_eu, true>;
            }
        }
        if constexpr (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S) {
            if (p.iq_signs) {
                kernel = mul_mat_vec_q_gcn_grp<type, c_ncols_dst, ngroups, nwarps, waves_per_eu, false, true>;
            }
        }
        kernel<<<block_nums, block_dims, 0, stream>>>(
            p.vx, p.vy, p.fusion, p.dst, p.ncols_x, p.nrows_x, p.stride_row_x, p.stride_col_y, p.stride_col_dst,
            p.channel_ratio, p.stride_channel_x, p.stride_channel_y, p.stride_channel_dst, p.sample_ratio,
            p.stride_sample_x, p.stride_sample_y, p.stride_sample_dst);
    }
}

template <ggml_type type, int c_ncols_dst, int waves_per_eu, bool gate, bool aligned_b2, bool planes = false>
static void mul_mat_vec_q_gcn_mc_switch_shape(const int ngroups, const int nwarps, const mmvq_gcn_mc_args & p, cudaStream_t stream) {
    if (ngroups == 2 && !gate) {
        switch (nwarps) {
            case 1:  mul_mat_vec_q_gcn_mc_launch<type, c_ncols_dst, 2, 1, waves_per_eu, false, aligned_b2, planes>(p, stream); break;
            case 2:  mul_mat_vec_q_gcn_mc_launch<type, c_ncols_dst, 2, 2, waves_per_eu, false, aligned_b2, planes>(p, stream); break;
            default: mul_mat_vec_q_gcn_mc_launch<type, c_ncols_dst, 2, 4, waves_per_eu, false, aligned_b2, planes>(p, stream); break;
        }
    } else {
        switch (nwarps) {
            case 1:  mul_mat_vec_q_gcn_mc_launch<type, c_ncols_dst, 4, 1, waves_per_eu, gate, aligned_b2, planes>(p, stream); break;
            case 2:  mul_mat_vec_q_gcn_mc_launch<type, c_ncols_dst, 4, 2, waves_per_eu, gate, aligned_b2, planes>(p, stream); break;
            default: mul_mat_vec_q_gcn_mc_launch<type, c_ncols_dst, 4, 4, waves_per_eu, gate, aligned_b2, planes>(p, stream); break;
        }
    }
}

template <ggml_type type, int c_ncols_dst, bool gate, bool aligned_b2, bool planes = false>
static void mul_mat_vec_q_gcn_mc_switch_wpe(const int waves_per_eu, const int ngroups, const int nwarps,
        const mmvq_gcn_mc_args & p, cudaStream_t stream) {
    if constexpr (mmvq_gcn_mc_wpe_alt(type, c_ncols_dst) != 0 && !gate) {
        if (waves_per_eu == mmvq_gcn_mc_wpe_alt(type, c_ncols_dst)) {
            mul_mat_vec_q_gcn_mc_switch_shape<type, c_ncols_dst, mmvq_gcn_mc_wpe_alt(type, c_ncols_dst), false, aligned_b2, planes>(
                ngroups, nwarps, p, stream);
            return;
        }
    }
    mul_mat_vec_q_gcn_mc_switch_shape<type, c_ncols_dst, mmvq_gcn_mc_wpe(type, c_ncols_dst), gate, aligned_b2, planes>(ngroups, nwarps, p, stream);
    GGML_UNUSED(waves_per_eu);
}

template <ggml_type type, bool gate, bool aligned_b2, bool planes = false>
static bool mul_mat_vec_q_gcn_mc_switch_ncols(const int ncols_dst, const int waves_per_eu, const int ngroups,
        const int nwarps, const mmvq_gcn_mc_args & p, cudaStream_t stream) {
    switch (ncols_dst) {
        case 1:
            // the Q6_K output head of the MTP drafts (248k rows), and the others on request
            if constexpr ((!gate || mmvq_gcn_mc_q4(type)) && mmvq_gcn_mc_type_env(type) == nullptr && !planes) {
                mul_mat_vec_q_gcn_mc_switch_wpe<type, 1, gate, aligned_b2, planes>(waves_per_eu, ngroups, nwarps, p, stream);
                return true;
            }
            return false;
        case 2:
            if constexpr (planes) {
                return false;
            }
            mul_mat_vec_q_gcn_mc_switch_wpe<type, 2, gate, aligned_b2, planes>(waves_per_eu, ngroups, nwarps, p, stream);
            return true;
        case 3: mul_mat_vec_q_gcn_mc_switch_wpe<type, 3, gate, aligned_b2, planes>(waves_per_eu, ngroups, nwarps, p, stream); return true;
        case 4: mul_mat_vec_q_gcn_mc_switch_wpe<type, 4, gate, aligned_b2, planes>(waves_per_eu, ngroups, nwarps, p, stream); return true;
        // the aligned Q4_0 loads lose at 5..8 columns (pp8 -9 %)
        case 5: mul_mat_vec_q_gcn_mc_switch_wpe<type, 5, gate, false, planes>(waves_per_eu, ngroups, nwarps, p, stream); return true;
        case 6: mul_mat_vec_q_gcn_mc_switch_wpe<type, 6, gate, false, planes>(waves_per_eu, ngroups, nwarps, p, stream); return true;
        case 7: mul_mat_vec_q_gcn_mc_switch_wpe<type, 7, gate, false, planes>(waves_per_eu, ngroups, nwarps, p, stream); return true;
        case 8: mul_mat_vec_q_gcn_mc_switch_wpe<type, 8, gate, false, planes>(waves_per_eu, ngroups, nwarps, p, stream); return true;
        default: return false;
    }
}

// A/B switches, 0 = old path: GGML_MMVQ_GCN_MC, _K (Q4_K/Q5_K/Q6_K), _Q4K (Q4_K), _Q8 (Q8_0), _Q2K, _Q3K, _IQ2S,
// _IQ3XXS, _IQ3S, _IQ4NL, _IQ4XS, _LDS (Q5_K/Q6_K activations through LDS), _Q6K1 (1-column Q6_K), _FUSED; _1COL=1
// adds the other 1-column calls of Q4_0/Q4_1/Q5_K; _GROUPS (2|4), _WARPS (1|2|4), _WPE (2|3) override the shapes chosen below.
template <ggml_type type>
static bool mul_mat_vec_q_gcn_mc_try(const int ncols_dst, const mmvq_gcn_mc_args & p, cudaStream_t stream) {
    constexpr const char * type_env = mmvq_gcn_mc_type_env(type);
    static const bool enabled     = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC", 1) != 0 &&
        (type_env ? mmvq_gcn_mc_env(type_env, 1) != 0 :
            mmvq_gcn_mc_q4(type) ||
            mmvq_gcn_mc_env(type == GGML_TYPE_Q8_0 ? "GGML_MMVQ_GCN_MC_Q8" : "GGML_MMVQ_GCN_MC_K", 1) != 0) &&
        (type != GGML_TYPE_Q4_K || mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_Q4K", 1) != 0);
    static const bool q6_k_1      = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_Q6K1", 1) != 0;
    static const bool one_col     = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_1COL", 0) != 0;
    static const bool fused       = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_FUSED", 1) != 0;
    static const int  env_ngroups = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_GROUPS", 0);
    static const int  env_nwarps  = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_WARPS", 0);
    static const int  env_wpe     = mmvq_gcn_mc_env("GGML_MMVQ_GCN_MC_WPE", 0);

    const ggml_cuda_mm_fusion_args_device & f = p.fusion;
    const bool has_fusion = f.gate || f.x_bias || f.gate_bias || f.x_scale || f.gate_scale || f.shared_up;
    const bool gate = f.gate != nullptr;
    if (!enabled || ncols_dst > 8 || p.nrows_x < mmvq_gcn_mc_rows(type, ncols_dst, gate) ||
            (ncols_dst == 1 && !(type == GGML_TYPE_Q6_K && !has_fusion ? q6_k_1 : one_col)) ||
            (has_fusion && (!fused || f.x_scale || f.gate_scale || f.shared_up || (gate && !mmvq_gcn_mc_q4(type))))) {
        return false;
    }

    const int waves_per_eu = env_wpe == 3 ? 3 : 2;
    const int ngroups      = env_ngroups == 2 && !gate ? 2 : 4;
    int nwarps = env_nwarps;
    if (nwarps != 1 && nwarps != 2 && nwarps != 4) {
        // enough waves to fill the card, but at least two K iterations per wave
        constexpr int lpb = mmvq_gcn_mc_whole(type) ? 1 : ggml_cuda_type_traits<type>::qi / get_vdr_mmvq(type);
        const int     rows        = mmvq_gcn_mc_rows(type, ncols_dst, gate);
        const int64_t nrow_groups = int64_t((p.nrows_x + rows*ngroups - 1) / (rows*ngroups)) * p.nchannels_dst * p.nsamples_dst;
        const int     nchunks_k   = p.ncols_x / ggml_blck_size(type) / ((64/ngroups)/lpb);
        nwarps = 1;
        while (nwarps < 4 && nrow_groups*nwarps < 2048 && nchunks_k >= 4*nwarps) {
            nwarps *= 2;
        }
    }

    if constexpr (type == GGML_TYPE_Q4_0) {
        static const bool aligned_b2 = mmvq_gcn_mc_env("GGML_MMVQ_ALIGNED_B2", 1) != 0;
        if (aligned_b2) {
            return gate ? mul_mat_vec_q_gcn_mc_switch_ncols<type, true, true>(ncols_dst, waves_per_eu, ngroups, nwarps, p, stream) :
                mul_mat_vec_q_gcn_mc_switch_ncols<type, false, true>(ncols_dst, waves_per_eu, ngroups, nwarps, p, stream);
        }
    }
    if constexpr (mmvq_gcn_mc_q4(type)) {
        if (gate) {
            return mul_mat_vec_q_gcn_mc_switch_ncols<type, true, false>(ncols_dst, waves_per_eu, ngroups, nwarps, p, stream);
        }
    }
    if constexpr (type == GGML_TYPE_Q8_0) {
        if (p.planes) {
            return mul_mat_vec_q_gcn_mc_switch_ncols<type, false, false, true>(ncols_dst, waves_per_eu, ngroups, nwarps, p, stream);
        }
    }
    return mul_mat_vec_q_gcn_mc_switch_ncols<type, false, false>(ncols_dst, waves_per_eu, ngroups, nwarps, p, stream);
}

struct mmvq_gcn_wide_args {
    const void * vx;
    const block_q8_1_mmq * vy;
    float * dst;
    int ncols_dst, ncols_x, nrows_x, stride_row_x, stride_col_dst;
    uint3 channel_ratio;
    int stride_channel_x, stride_channel_y, stride_channel_dst;
    uint3 sample_ratio;
    int stride_sample_x, stride_sample_y, stride_sample_dst;
    int nchannels_dst, nsamples_dst;
};

template <ggml_type type, int nwarps, int nb>
static void mul_mat_vec_q_gcn_wide_launch(const mmvq_gcn_wide_args & p, cudaStream_t stream) {
    const dim3 block_nums((p.nrows_x + 63)/64, p.nchannels_dst, p.nsamples_dst);
    const dim3 block_dims(64, nwarps, 1);
    const auto kernel = p.ncols_dst <= 9 ? mul_mat_vec_q_gcn_wide<type, 9, nwarps, nb> :
        p.ncols_dst <= 12 ? mul_mat_vec_q_gcn_wide<type, 12, nwarps, nb> : mul_mat_vec_q_gcn_wide<type, 16, nwarps, nb>;
    kernel<<<block_nums, block_dims, 0, stream>>>(
        p.vx, p.vy, p.dst, p.ncols_dst, p.ncols_x, p.nrows_x, p.stride_row_x, p.stride_col_dst,
        p.channel_ratio, p.stride_channel_x, p.stride_channel_y, p.stride_channel_dst, p.sample_ratio,
        p.stride_sample_x, p.stride_sample_y, p.stride_sample_dst);
}

template <ggml_type type, int nb>
static void mul_mat_vec_q_gcn_wide_switch_warps(const int nwarps, const mmvq_gcn_wide_args & p, cudaStream_t stream) {
    switch (nwarps) {
        case 1:  mul_mat_vec_q_gcn_wide_launch<type,  1, nb>(p, stream); break;
        case 2:  mul_mat_vec_q_gcn_wide_launch<type,  2, nb>(p, stream); break;
        case 4:  mul_mat_vec_q_gcn_wide_launch<type,  4, nb>(p, stream); break;
        case 8:  mul_mat_vec_q_gcn_wide_launch<type,  8, nb>(p, stream); break;
        default: mul_mat_vec_q_gcn_wide_launch<type, 16, nb>(p, stream); break;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_gcn_wide_switch(const int nb, const int nwarps, const mmvq_gcn_wide_args & p, cudaStream_t stream) {
    if (nb == 1) {
        mul_mat_vec_q_gcn_wide_switch_warps<type, 1>(nwarps, p, stream);
    } else {
        mul_mat_vec_q_gcn_wide_switch_warps<type, 2>(nwarps, p, stream);
    }
}

struct mmvq_gcn_wide_shape {
    int nb, nwarps;
};

// _NB (1|2, blocks per lane and step) and _WARPS (1|2|4|8|16) override the shape.
static mmvq_gcn_wide_shape mmvq_gcn_wide_get_shape(const ggml_type type, const int64_t nrow_groups, const int64_t blocks_per_row) {
    static const int env_nb     = mmvq_gcn_mc_env("GGML_MMVQ_GCN_WIDE_NB", 0);
    static const int env_nwarps = mmvq_gcn_mc_env("GGML_MMVQ_GCN_WIDE_WARPS", 0);

    // measured on the Qwen3.8 27B shapes: Q8_0 one block per step and 16 waves, Q4 two blocks and 8 waves (16 below 40 row groups)
    mmvq_gcn_wide_shape sh;
    sh.nb = env_nb == 1 || env_nb == 2 ? env_nb : (type == GGML_TYPE_Q8_0 ? 1 : 2);
    sh.nwarps = env_nwarps;
    if (sh.nwarps != 1 && sh.nwarps != 2 && sh.nwarps != 4 && sh.nwarps != 8 && sh.nwarps != 16) {
        sh.nwarps = type == GGML_TYPE_Q8_0 || nrow_groups < 40 ? 16 : 8;
        while (sh.nwarps > 1 && blocks_per_row/sh.nb < 2*sh.nwarps) {
            sh.nwarps /= 2;
        }
    }
    return sh;
}

static void mul_mat_vec_q_gcn_wide_cuda(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    GGML_ASSERT(GGML_CUDA_CC_IS_GCN(cc) && mmvq_gcn_wide_type(src0->type) && ne11 <= MMVQ_GCN_WIDE_MAX_BATCH);

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    // the kernel reads all columns of the last chunk, also those past ne11
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  nbytes_y    = ne13*ne12*ne11*ne10_padded*sizeof(block_q8_1)/QK8_1 + MMVQ_GCN_WIDE_MAX_BATCH*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_y);
    quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), src0->type, ne10,
        nb11/ts_src1, nb12/ts_src1, nb13/ts_src1, ne10_padded, ne11, ne12, ne13, stream);

    const int64_t nrow_groups = (ne01 + 63)/64;
    const mmvq_gcn_wide_shape sh = mmvq_gcn_wide_get_shape(src0->type, nrow_groups*ne2*ne3, ne00/ggml_blck_size(src0->type));
    const int stride_channel_y = ne11*ne10_padded/QK8_1_MMQ;
    const mmvq_gcn_wide_args p = {
        src0->data, (const block_q8_1_mmq *) src1_q8_1.get(), (float *) dst->data,
        int(ne11), int(ne00), int(ne01), int(nb01/ts_src0), int(nb1/ts_dst),
        init_fastdiv_values(ne2/ne02), int(nb02/ts_src0), stride_channel_y, int(nb2/ts_dst),
        init_fastdiv_values(ne3/ne03), int(nb03/ts_src0), int(ne12*stride_channel_y), int(nb3/ts_dst),
        int(ne2), int(ne3),
    };
    switch (src0->type) {
        case GGML_TYPE_Q4_0: mul_mat_vec_q_gcn_wide_switch<GGML_TYPE_Q4_0>(sh.nb, sh.nwarps, p, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_vec_q_gcn_wide_switch<GGML_TYPE_Q4_1>(sh.nb, sh.nwarps, p, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_vec_q_gcn_wide_switch<GGML_TYPE_Q8_0>(sh.nb, sh.nwarps, p, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

// Q8_0 planes, see q8_0_planes_qs: ushort u of a group in planes is ushort q8_0_planes_src(u) of the canonical group.
static __device__ __forceinline__ int q8_0_planes_src(const int u, const bool to_planes) {
    if (to_planes) {
        return u < 128 ? 17*(u >> 4) + 1 + (u & 15) : 17*(u - 128);
    }
    const int j = u / 17;
    const int r = u - 17*j;
    return r == 0 ? 128 + j : 16*j + r - 1;
}

// src may be dst: each block reads its groups completely before it writes them.
template <bool to_planes>
static __global__ void q8_0_planes_convert(const char * src, char * dst, const int64_t ngroups) {
    constexpr int ngb = 32;
    constexpr int nw  = sizeof(block_q8_0)*8/4;
    __shared__ uint16_t buf[ngb*nw*2];

    const int64_t g0 = int64_t(blockIdx.x)*ngb;
    const int     ng = int(min(int64_t(ngb), ngroups - g0));
    const uint32_t * s = (const uint32_t *) (src + g0*nw*4);
    uint32_t       * d = (uint32_t       *) (dst + g0*nw*4);

    for (int i = threadIdx.x; i < ng*nw; i += blockDim.x) {
        ((uint32_t *) buf)[i] = s[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < ng*nw; i += blockDim.x) {
        const uint16_t * bg = buf + (i/nw)*nw*2;
        const int        u  = 2*(i % nw);
        d[i] = uint32_t(bg[q8_0_planes_src(u, to_planes)]) | uint32_t(bg[q8_0_planes_src(u + 1, to_planes)]) << 16;
    }
}

static void q8_0_planes_convert_cuda(const char * src, char * dst, const int64_t ngroups, const bool to_planes, cudaStream_t stream) {
    if (ngroups <= 0) {
        return;
    }
    const int64_t nblocks = (ngroups + 31) / 32;
    if (to_planes) {
        q8_0_planes_convert<true><<<nblocks, 256, 0, stream>>>(src, dst, ngroups);
    } else {
        q8_0_planes_convert<false><<<nblocks, 256, 0, stream>>>(src, dst, ngroups);
    }
    CUDA_CHECK(cudaGetLastError());
}

// GGML_CUDA_Q8_0_PLANES: 0 off, 1 2D Q8_0 weights, 2 every Q8_0 tensor it can take (for test-backend-ops)
static int q8_0_planes_mode() {
    static const int mode = mmvq_gcn_mc_env("GGML_CUDA_Q8_0_PLANES", 1);
    return mode;
}

static std::mutex                           q8_0_planes_mutex;
static std::map<const char *, const char *> q8_0_planes_ranges;
static std::atomic<int>                     q8_0_planes_count{0};

static bool q8_0_planes_find(const void * ptr, const char ** lo) {
    if (q8_0_planes_count.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(q8_0_planes_mutex);
    auto it = q8_0_planes_ranges.upper_bound((const char *) ptr);
    if (it == q8_0_planes_ranges.begin()) {
        return false;
    }
    --it;
    if ((const char *) ptr >= it->second) {
        return false;
    }
    *lo = it->first;
    return true;
}

// Reads or writes [ptr, ptr + size) of a tensor in planes in the canonical layout, through whole groups.
static void q8_0_planes_rw(const char * lo, char * ptr, void * host, const size_t size, const bool write, cudaStream_t stream) {
    if (size == 0) {
        return;
    }
    constexpr size_t gs = 8*sizeof(block_q8_0);
    const size_t off = ptr - lo;
    const size_t g0  = off / gs;
    const size_t g1  = (off + size + gs - 1) / gs;
    char * groups = (char *) lo + g0*gs;
    char * tmp    = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &tmp, (g1 - g0)*gs));
    q8_0_planes_convert_cuda(groups, tmp, g1 - g0, false, stream);
    if (write) {
        CUDA_CHECK(cudaMemcpyAsync(tmp + (off - g0*gs), host, size, cudaMemcpyHostToDevice, stream));
        q8_0_planes_convert_cuda(tmp, groups, g1 - g0, true, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(host, tmp + (off - g0*gs), size, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));
}
#endif // defined(GGML_USE_HIP)

bool ggml_cuda_q8_0_planes_eligible(const ggml_tensor * tensor, const int device) {
#if defined(GGML_USE_HIP)
    const int mode = q8_0_planes_mode();
    if (mode == 0 || tensor->type != GGML_TYPE_Q8_0 || tensor->view_src != nullptr || !ggml_is_contiguous(tensor) ||
            tensor->ne[0] % (8*QK8_0) != 0 || (uintptr_t) tensor->data % 16 != 0 || tensor->buffer == nullptr ||
            !GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    const ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(tensor->buffer);
    if (mode == 2) {
        return usage != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    }
    // the KV cache is in a weights buffer too
    const size_t nlen = strlen(tensor->name);
    // the 48-row ssm_alpha/ssm_beta stay canonical for the fused GDN gate (ggml_cuda_should_use_mmvq_gdn_gate)
    return usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && tensor->ne[2] == 1 && tensor->ne[3] == 1 && tensor->ne[1] > 64 &&
        nlen > 7 && strcmp(tensor->name + nlen - 7, ".weight") == 0;
#else
    GGML_UNUSED_VARS(tensor, device);
    return false;
#endif // defined(GGML_USE_HIP)
}

void ggml_cuda_q8_0_planes_repack(const ggml_tensor * tensor, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    const char * lo = (const char *) tensor->data;
    const size_t nbytes = ggml_nbytes(tensor);
    q8_0_planes_convert_cuda(lo, (char *) lo, nbytes / (8*sizeof(block_q8_0)), true, stream);

    std::lock_guard<std::mutex> lock(q8_0_planes_mutex);
    q8_0_planes_ranges[lo] = lo + nbytes;
    q8_0_planes_count = (int) q8_0_planes_ranges.size();
#else
    GGML_UNUSED_VARS(tensor, stream);
#endif // defined(GGML_USE_HIP)
}

bool ggml_cuda_q8_0_planes_get(const ggml_tensor * tensor, void * data, const size_t offset, const size_t size, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    const char * lo;
    if (!q8_0_planes_find(tensor->data, &lo)) {
        return false;
    }
    q8_0_planes_rw(lo, (char *) tensor->data + offset, data, size, false, stream);
    return true;
#else
    GGML_UNUSED_VARS(tensor, data, offset, size, stream);
    return false;
#endif // defined(GGML_USE_HIP)
}

bool ggml_cuda_q8_0_planes_set(const ggml_tensor * tensor, const void * data, const size_t offset, const size_t size, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    const char * lo;
    if (!q8_0_planes_find(tensor->data, &lo) || (tensor->view_src == nullptr && offset == 0 && size == ggml_nbytes(tensor))) {
        return false;
    }
    q8_0_planes_rw(lo, (char *) tensor->data + offset, const_cast<void *>(data), size, true, stream);
    return true;
#else
    GGML_UNUSED_VARS(tensor, data, offset, size, stream);
    return false;
#endif // defined(GGML_USE_HIP)
}

void ggml_cuda_q8_0_planes_forget(const void * base, const size_t size) {
#if defined(GGML_USE_HIP)
    if (q8_0_planes_count.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(q8_0_planes_mutex);
    const char * lo = (const char *) base;
    q8_0_planes_ranges.erase(q8_0_planes_ranges.lower_bound(lo), q8_0_planes_ranges.lower_bound(lo + size));
    q8_0_planes_count = (int) q8_0_planes_ranges.size();
#else
    GGML_UNUSED_VARS(base, size);
#endif // defined(GGML_USE_HIP)
}

bool ggml_cuda_q8_0_planes(const ggml_tensor * tensor) {
#if defined(GGML_USE_HIP)
    const char * lo;
    return tensor->type == GGML_TYPE_Q8_0 && q8_0_planes_find(tensor->data, &lo);
#else
    GGML_UNUSED(tensor);
    return false;
#endif // defined(GGML_USE_HIP)
}

void ggml_cuda_q8_0_planes_unpack(const ggml_tensor * tensor, void * dst, cudaStream_t stream) {
#if defined(GGML_USE_HIP)
    const char * lo;
    GGML_ASSERT(q8_0_planes_find(tensor->data, &lo) && ((const char *) tensor->data - lo) % (8*sizeof(block_q8_0)) == 0);
    const int64_t ngroups = (ggml_nbytes(tensor) + 8*sizeof(block_q8_0) - 1) / (8*sizeof(block_q8_0));
    q8_0_planes_convert_cuda((const char *) tensor->data, (char *) dst, ngroups, false, stream);
#else
    GGML_UNUSED_VARS(tensor, dst, stream);
    GGML_ABORT("Q8_0 planes need HIP");
#endif // defined(GGML_USE_HIP)
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool aligned_b2 = false, bool planes = false, bool iq_signs = false, bool grid_lds = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream, const bool kvdr4 = false) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
#if defined(GGML_USE_HIP)
    // From 3 columns on the generic kernel needs 86+ VGPRs with VDR 4 (2 waves per SIMD) and is slower than with VDR 2.
    if constexpr (mmvq_gcn_kvdr4_type(type) && c_ncols_dst <= 2) {
        if (kvdr4) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            if constexpr (c_ncols_dst == 1) {
                if (has_fusion) {
                    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, false, false, false, false, true>, launch_params,
                         vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                         channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                         sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
                    return;
                }
            }
            GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, false, false, false, false, true>, launch_params,
                vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }
#else
    GGML_UNUSED(kvdr4);
#endif // defined(GGML_USE_HIP)
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, aligned_b2, planes, iq_signs, grid_lds>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, aligned_b2, planes, iq_signs, grid_lds>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

template <ggml_type type, bool aligned_b2 = false, bool planes = false, bool iq_signs = false, bool grid_lds = false>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

#if defined(GGML_USE_HIP)
    if constexpr (!grid_lds && ggml_cuda_iq_grid_lds_type(type)) {
        static const bool use_grid_lds = ggml_cuda_iq_grid_lds_enabled("GGML_MMVQ_IQ_GRID_LDS", type);
        if (use_grid_lds && GGML_CUDA_CC_IS_GCN(cc)) {
            mul_mat_vec_q_switch_ncols_dst<type, aligned_b2, planes, iq_signs, true>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    // GGML_MMVQ_IQ_SIGNS: up to how many columns IQ2_S, IQ3_XXS and IQ3_S take the dots without byte-wise negate,
    // 0 = old path; 7 and 8 columns are 2-5 % slower with them on the MI50
    if constexpr (!iq_signs && (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S)) {
        static const int iq_signs_max = mmvq_gcn_mc_env("GGML_MMVQ_IQ_SIGNS", 6);
        if (ncols_dst <= iq_signs_max && GGML_CUDA_CC_IS_GCN(cc)) {
            mul_mat_vec_q_switch_ncols_dst<type, false, false, true, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    // Q8_0 at 3..8 columns reads more bytes through L1 that way and gets slower; Q4_0 has the row-group kernel there.
    if constexpr ((type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) && !aligned_b2 && !planes) {
        static const bool use_aligned_b2 = mmvq_gcn_mc_env("GGML_MMVQ_ALIGNED_B2", 1) != 0;
        if (use_aligned_b2 && GGML_CUDA_CC_IS_GCN(cc) && ncols_dst <= 2) {
            mul_mat_vec_q_switch_ncols_dst<type, true>(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst,
                stride_row_x, stride_col_y, stride_col_dst, nchannels_x, nchannels_y, nchannels_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x, nsamples_dst,
                stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    const bool has_ids = ids != nullptr;

#if defined(GGML_USE_HIP)
    static const bool kvdr4_env = mmvq_gcn_kvdr4_type(type) && mmvq_gcn_mc_env("GGML_MMVQ_GCN_KVDR4", 1) != 0;
    const bool kvdr4 = kvdr4_env && table_id == MMVQ_PARAMETERS_GCN;
#else
    const bool kvdr4 = false;
#endif // defined(GGML_USE_HIP)

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    const int     vdr                   = kvdr4 ? VDR_Q4_K_Q5_K_Q8_1_MMVQ_GCN : get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

#if defined(GGML_USE_HIP)
    if constexpr (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_K ||
            type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || mmvq_gcn_mc_type_env(type) != nullptr) {
        if (GGML_CUDA_CC_IS_GCN(cc) && !has_ids) {
            const mmvq_gcn_mc_args p = {
                vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
                channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, nchannels_dst, nsamples_dst, planes, iq_signs,
            };
            if (mul_mat_vec_q_gcn_mc_try<type>(ncols_dst, p, stream)) {
                return;
            }
        }
    }
#endif // defined(GGML_USE_HIP)

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters, aligned_b2, planes, iq_signs, grid_lds>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream, kvdr4);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, aligned_b2, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, false, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, false, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, false, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, false, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, false, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, false, planes, iq_signs, grid_lds>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream, kvdr4);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// Only the graph's own src1 of the node being evaluated is cached: ops that call mat-vecs on temporary tensors
// could hand over a different tensor at the same address.
static bool q8_1_cache_usable(const ggml_backend_cuda_context & ctx, const ggml_tensor * src1) {
#if defined(GGML_USE_VMM)
    // the VMM pool frees in reverse allocation order only
    GGML_UNUSED_VARS(ctx, src1);
    return false;
#else
    return ctx.cur_cgraph != nullptr && ctx.cur_node_idx >= 0 && ctx.cur_node_idx < ctx.cur_cgraph->n_nodes &&
           ctx.cur_cgraph->nodes[ctx.cur_node_idx]->src[1] == src1;
#endif // defined(GGML_USE_VMM)
}

static bool q8_1_cache_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t size) {
    const auto & c = ctx.q8_1_cache;
    if (!c.buf || c.src1 != src1 || c.data != src1->data || c.size != size || c.stream != ctx.stream() ||
            !q8_1_cache_usable(ctx, src1) || c.node_idx > ctx.cur_node_idx) {
        return false;
    }
    // no node between the two mat-vecs may have written into src1
    const char * lo = (const char *) src1->data;
    const char * hi = lo + ggml_nbytes(src1);
    for (int k = c.node_idx + 1; k < ctx.cur_node_idx; ++k) {
        const ggml_tensor * t = ctx.cur_cgraph->nodes[k];
        if (ggml_op_is_empty(t->op) || ggml_is_empty(t) || t->data == nullptr) {
            continue;
        }
        const char * t_lo = (const char *) t->data;
        const char * t_hi = t_lo + ggml_nbytes(t);
        if (t_lo < hi && lo < t_hi) {
            return false;
        }
    }
    return true;
}

static char * q8_1_cache_store(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const size_t size) {
    auto & c = ctx.q8_1_cache;
    c.reset();
    c.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), size);
    c.src1     = src1;
    c.data     = src1->data;
    c.node_idx = ctx.cur_node_idx;
    c.stream   = ctx.stream();
    c.size     = size;
    return c.buf->get();
}

char * ggml_cuda_q8_1_cache_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
#if defined(GGML_USE_VMM)
    GGML_UNUSED_VARS(ctx, cgraph, node_idx);
    return nullptr;
#else
    const ggml_tensor * t = cgraph->nodes[node_idx];
    if (ctx.cur_cgraph != cgraph || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return nullptr;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * src1 = nullptr;
    for (int j = node_idx + 1; j < cgraph->n_nodes && j <= node_idx + 64 && src1 == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op != GGML_OP_MUL_MAT || !ggml_is_quantized(n->src[0]->type)) {
            continue;
        }
        // the mat-vec may read t through a reshape (the GDN output projection does)
        const ggml_tensor * y = n->src[1];
        if ((y == t || (y->op == GGML_OP_RESHAPE && y->src[0] == t)) && ggml_is_contiguous(y) &&
                y->ne[0] % MATRIX_ROW_PADDING == 0 && y->ne[1] <= MMVQ_MAX_BATCH_SIZE && y->ne[2] == 1 && y->ne[3] == 1 &&
                ggml_cuda_should_use_mmvq(n->src[0]->type, cc, y->ne[1], n->src[0]->ne[1])) {
            src1 = y;
        }
    }
    if (src1 == nullptr) {
        return nullptr;
    }
    const size_t size = src1->ne[1] * (src1->ne[0] / QK8_1) * sizeof(block_q8_1);
    auto & c = ctx.q8_1_cache;
    c.reset();
    c.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), size);
    c.src1     = src1;
    c.data     = src1->data;
    c.node_idx = node_idx;
    c.stream   = ctx.stream();
    c.size     = size;
    return c.buf->get();
#endif // defined(GGML_USE_VMM)
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

#if defined(GGML_USE_HIP)
    // only the dense kernels read Q8_0 planes; the others get canonical copies
    {
        const ggml_tensor * w[4] = { src0, fusion ? fusion->gate : nullptr, fusion ? fusion->shared_up : nullptr, fusion ? fusion->shared_gate : nullptr };
        bool p[4];
        for (int k = 0; k < 4; ++k) {
            p[k] = w[k] && ggml_cuda_q8_0_planes(w[k]);
        }
        const bool dense = !ids && !w[2];
        if ((!dense && (p[0] || p[1] || p[2] || p[3])) || (w[1] && p[0] != p[1])) {
            ggml_cuda_pool_alloc<char> buf[4];
            ggml_tensor                canonical[4];
            for (int k = 0; k < 4; ++k) {
                if (p[k]) {
                    buf[k].alloc(ctx.pool(), GGML_PAD(ggml_nbytes(w[k]), 8*sizeof(block_q8_0)));
                    ggml_cuda_q8_0_planes_unpack(w[k], buf[k].get(), stream);
                    canonical[k]      = *w[k];
                    canonical[k].data = buf[k].get();
                    w[k]              = &canonical[k];
                }
            }
            if (fusion) {
                ggml_cuda_mm_fusion_args_host f = *fusion;
                f.gate        = w[1];
                f.shared_up   = w[2];
                f.shared_gate = w[3];
                ggml_cuda_mul_mat_vec_q(ctx, w[0], src1, ids, dst, &f);
            } else {
                ggml_cuda_mul_mat_vec_q(ctx, w[0], src1, ids, dst);
            }
            return;
        }
    }
#endif // defined(GGML_USE_HIP)

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

#if defined(GGML_USE_HIP)
    if (!ids && ne11 > MMVQ_MAX_BATCH_SIZE) {
        GGML_ASSERT(!fusion);
        mul_mat_vec_q_gcn_wide_cuda(ctx, src0, src1, dst);
        return;
    }
#endif // defined(GGML_USE_HIP)

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  nbytes_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1_tmp(ctx.pool());
    char * src1_q8_1 = nullptr;
    if (!ids && q8_1_cache_lookup(ctx, src1, nbytes_q8_1)) {
        src1_q8_1 = ctx.q8_1_cache.buf->get();
    } else {
        if (!ids && q8_1_cache_usable(ctx, src1)) {
            src1_q8_1 = q8_1_cache_store(ctx, src1, nbytes_q8_1);
        } else {
            src1_q8_1 = src1_q8_1_tmp.alloc(nbytes_q8_1);
        }
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

#if defined(GGML_USE_HIP)
    if (ggml_cuda_q8_0_planes(src0)) {
        mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0, false, true>(
            src0->data, src1_q8_1, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
            ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
            ne03, ne3, s03, s13, s3, ids_stride, stream);
        return;
    }
#endif // defined(GGML_USE_HIP)

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

// Qwen3.5/3.8 GDN gate with quantized ssm_alpha/ssm_beta: grid rows past the first matrix compute beta from the same y.
// K loop and reduction are those of mul_mat_vec_q with the GCN table's nwarps, so both projections round as unfused.
template <ggml_type type, int ncols_dst, int nwarps, int rows_per_block>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_gdn_gate(
        const void * __restrict__ vx_a, const void * __restrict__ vx_b, const block_q8_1 * __restrict__ y,
        const float * __restrict__ dt, const float * __restrict__ a, float * __restrict__ dst_a, float * __restrict__ dst_b,
        const int ncols_x, const int nrows_x, const int ncols_y, const int stride_row_x, const int stride_col_y) {
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const int  tid         = warp_size*threadIdx.y + threadIdx.x;
    const int  nrow_blocks = (nrows_x + rows_per_block - 1)/rows_per_block;
    const bool part_b      = (int) blockIdx.x >= nrow_blocks;
    const int  row0        = (part_b ? blockIdx.x - nrow_blocks : blockIdx.x)*rows_per_block;
    const int  col0        = blockIdx.y*ncols_dst;
    const void * vx        = part_b ? vx_b : vx_a;

    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter  = vdr * nwarps*warp_size / qi;

    // Tail rows and columns read the last valid one and are not written.
    int kbx_row[rows_per_block];
#pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        kbx_row[i] = min(row0 + i, nrows_x - 1)*stride_row_x;
    }
    const block_q8_1 * y_col[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        y_col[j] = y + min(col0 + j, ncols_y - 1)*stride_col_y;
    }

    float tmp[ncols_dst][rows_per_block] = {{0.0f}};

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y_col[j][kby], kbx_row[i] + kbx, kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_block][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // Every lane holds all sums after the reduction; lane j*rows_per_block + i takes one, so the epilogue runs once per wave.
    static_assert(ncols_dst*rows_per_block <= warp_size, "one output per lane");
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if ((int) threadIdx.x == j*rows_per_block + i) {
                sum = tmp[j][i];
            }
        }
    }

    const int row = row0 + threadIdx.x % rows_per_block;
    const int col = col0 + threadIdx.x / rows_per_block;
    if ((int) threadIdx.x >= ncols_dst*rows_per_block || row >= nrows_x || col >= ncols_y) {
        return;
    }
    // The expressions of op_sigmoid and op_softplus in unary.cu, so the gate matches the unfused graph.
    if (part_b) {
        dst_b[col*nrows_x + row] = 1.0f / (1.0f + expf(-sum));
    } else {
        const float v = sum + dt[row];
        dst_a[col*nrows_x + row] = ((v > 20.0f) ? v : log1pf(expf(v))) * a[row];
    }
}

struct mmvq_gdn_gate_args {
    const void       * x_a;
    const void       * x_b;
    const block_q8_1 * y;
    const float      * dt;
    const float      * a;
    float            * dst_a;
    float            * dst_b;
    int                ncols_x;
    int                nrows_x;
    int                ncols_y;
    int                stride_row_x;
    int                stride_col_y;
};

template <ggml_type type, int ncols_dst, int nwarps>
static void mul_mat_vec_q_gdn_gate_launch(const mmvq_gdn_gate_args & p, const int rows_per_block, const int warp_size, cudaStream_t stream) {
    const dim3 block_dims(warp_size, nwarps, 1);
    const int  nblocks_y = (p.ncols_y + ncols_dst - 1)/ncols_dst;

#define LAUNCH(rpb_) \
    mul_mat_vec_q_gdn_gate<type, ncols_dst, nwarps, rpb_><<<dim3(2*((p.nrows_x + rpb_ - 1)/rpb_), nblocks_y, 1), block_dims, 0, stream>>> \
        (p.x_a, p.x_b, p.y, p.dt, p.a, p.dst_a, p.dst_b, p.ncols_x, p.nrows_x, p.ncols_y, p.stride_row_x, p.stride_col_y)
    switch (rows_per_block) {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 4: LAUNCH(4); break;
        default: GGML_ABORT("fatal error");
    }
#undef LAUNCH
}

static int mmvq_gdn_gate_env_int(const char * name, const int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

static int64_t mmvq_gdn_gate_max_ncols() {
    // GGML_Q8_GATE_MAXN bounds the column counts taken; 9+ columns otherwise run two MMQ calls with a 128-row tile.
    static const int64_t max_n = std::min(mmvq_gdn_gate_env_int("GGML_Q8_GATE_MAXN", 16), 2*MMVQ_MAX_BATCH_SIZE);
    return max_n;
}

bool ggml_cuda_should_use_mmvq_gdn_gate(const ggml_tensor * mm_a, const ggml_tensor * mm_b, int cc) {
    // GGML_Q8_GATE=0 keeps the GDN gate unfused for Q8_0 projections.
    static const bool enabled = mmvq_gdn_gate_env_int("GGML_Q8_GATE", 1) != 0;

    const ggml_tensor * x_a = mm_a->src[0];
    const ggml_tensor * x_b = mm_b->src[0];
    const ggml_tensor * y   = mm_a->src[1];

    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc) || y != mm_b->src[1]) {
        return false;
    }
    if (x_a->type != GGML_TYPE_Q8_0 || x_b->type != GGML_TYPE_Q8_0 || y->type != GGML_TYPE_F32) {
        return false;
    }
    // the kernel reads the canonical Q8_0 layout
    if (ggml_cuda_q8_0_planes(x_a) || ggml_cuda_q8_0_planes(x_b)) {
        return false;
    }
    if (!ggml_are_same_shape(x_a, x_b) || x_a->nb[1] != x_b->nb[1] || !ggml_is_contiguous(x_a) || !ggml_is_contiguous(x_b) ||
            x_a->ne[2] != 1 || x_a->ne[3] != 1 || x_a->ne[1] > 1024) {
        return false;
    }
    if (!ggml_is_contiguous(y) || y->ne[0] != x_a->ne[0] || y->ne[0] % MATRIX_ROW_PADDING != 0) {
        return false;
    }
    const int64_t n = ggml_nrows(y);
    return n >= 1 && n <= mmvq_gdn_gate_max_ncols() && ggml_nrows(mm_a) == n && ggml_nrows(mm_b) == n;
}

void ggml_cuda_mul_mat_vec_q_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * mm_b,
        const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * dst_a, ggml_tensor * dst_b) {
    const ggml_tensor * x_a = mm_a->src[0];
    const ggml_tensor * y   = mm_a->src[1];

    const int64_t ncols_x = x_a->ne[0];
    const int64_t nrows_x = x_a->ne[1];
    const int64_t ncols_y = ggml_nrows(y);

    GGML_ASSERT(ggml_is_contiguous(dst_a) && ggml_nelements(dst_a) == nrows_x*ncols_y);
    GGML_ASSERT(ggml_is_contiguous(dst_b) && ggml_nelements(dst_b) == nrows_x*ncols_y);

    cudaStream_t stream = ctx.stream();

    // the q8_1 copy the qkv mat-vec or the rms_norm before it left, as ggml_cuda_mul_mat_vec_q finds it
    const size_t nbytes_q8_1 = ncols_y*ncols_x*sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> y_q8_1_tmp(ctx.pool());
    char * y_q8_1 = nullptr;
    if (q8_1_cache_lookup(ctx, y, nbytes_q8_1)) {
        y_q8_1 = ctx.q8_1_cache.buf->get();
    } else {
        y_q8_1 = q8_1_cache_usable(ctx, y) ? q8_1_cache_store(ctx, y, nbytes_q8_1) : y_q8_1_tmp.alloc(nbytes_q8_1);
        quantize_row_q8_1_cuda((const float *) y->data, nullptr, y_q8_1, x_a->type, ncols_x, ncols_x, ncols_x*ncols_y,
            ncols_x*ncols_y, ncols_x, ncols_y, 1, 1, stream);
    }

    mmvq_gdn_gate_args p;
    p.x_a          = x_a->data;
    p.x_b          = mm_b->src[0]->data;
    p.y            = (const block_q8_1 *) y_q8_1;
    p.dt           = (const float *) dt->data;
    p.a            = (const float *) a->data;
    p.dst_a        = (float *) dst_a->data;
    p.dst_b        = (float *) dst_b->data;
    p.ncols_x      = ncols_x;
    p.nrows_x      = nrows_x;
    p.ncols_y      = ncols_y;
    p.stride_row_x = x_a->nb[1] / ggml_type_size(x_a->type);
    p.stride_col_y = ncols_x / QK8_1;

    // GGML_Q8_GATE_ROWS=1|2|4 sets the rows per block; mul_mat_vec_q's GCN table uses 1 at one column, else 2.
    static const int rows_env = mmvq_gdn_gate_env_int("GGML_Q8_GATE_ROWS", 0);
    const int rows_per_block = rows_env == 1 || rows_env == 2 || rows_env == 4 ? rows_env : (ncols_y == 1 ? 1 : 2);

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    constexpr ggml_type type = GGML_TYPE_Q8_0;
    switch (ncols_y) {
        case 1: mul_mat_vec_q_gdn_gate_launch<type, 1, calc_nwarps(type, 1, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 2: mul_mat_vec_q_gdn_gate_launch<type, 2, calc_nwarps(type, 2, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 3: mul_mat_vec_q_gdn_gate_launch<type, 3, calc_nwarps(type, 3, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 4: mul_mat_vec_q_gdn_gate_launch<type, 4, calc_nwarps(type, 4, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 5: mul_mat_vec_q_gdn_gate_launch<type, 5, calc_nwarps(type, 5, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 6: mul_mat_vec_q_gdn_gate_launch<type, 6, calc_nwarps(type, 6, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 7: mul_mat_vec_q_gdn_gate_launch<type, 7, calc_nwarps(type, 7, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        case 8: mul_mat_vec_q_gdn_gate_launch<type, 8, calc_nwarps(type, 8, MMVQ_PARAMETERS_GCN)>(p, rows_per_block, warp_size, stream); break;
        default:
            GGML_ASSERT(ncols_y <= 2*MMVQ_MAX_BATCH_SIZE);
            mul_mat_vec_q_gdn_gate_launch<type, MMVQ_MAX_BATCH_SIZE, 2>(p, rows_per_block, warp_size, stream);
            break;
    }
}
