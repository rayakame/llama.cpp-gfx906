#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// SET_ROWS whose row indices are cast(scale*arange(start, step) + bias) taken from element offset on.
struct set_rows_arange_idx {
    float   start  = 0.0f;
    float   step   = 0.0f;
    float   scale  = 0.0f;
    float   bias   = 0.0f;
    int64_t offset = 0;
};

void ggml_cuda_op_set_rows_arange(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const set_rows_arange_idx & ar);
