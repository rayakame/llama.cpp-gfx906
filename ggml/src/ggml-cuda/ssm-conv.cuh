#include "common.cuh"

#define GGML_CUDA_CONV_STEP_MAX_T   16
#define GGML_CUDA_CONV_STEP_MAX_WIN 8

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// silu(ssm_conv(concat(state, x))) with d_conv 4 for one sequence: the old state is read from the cache row rows[0],
// the new state windows are stored, win[r] gets the one that ends r columns before the last. Strides in floats.
struct ggml_cuda_conv_step_args {
    const float *   states;
    const int32_t * rows;
    int64_t         row_stride;
    const float *   x;
    int             x_st;
    int             x_sc;
    const float *   w;
    int             w_sc;
    float *         dst;
    int             dst_st;
    int             n_t;
    int             nr;
    float *         win[GGML_CUDA_CONV_STEP_MAX_WIN];
    int             n_win;
};

void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_step_args & args);
