#pragma once

#include "common.cuh"

// Q8_0 weights in planes (see ggml_cuda_q8_0_planes) times a src1 of at least this many columns, already quantized
// to the MMQ q8_1 layout with ds layout D4.
bool ggml_cuda_q8_0_planes_gemm_use(int64_t ne11);

// One 2D slice: w is ne01 rows of nb01 bytes in planes, xq the slice's ne11 columns of block_q8_1_mmq
// (ne00/128 groups of ne11 blocks), dst ne11 rows of ne01 floats dst_s1 apart.
void ggml_cuda_q8_0_planes_gemm(const char * w, const void * xq, float * dst, int64_t ne00, int64_t ne01, size_t nb01,
                                int64_t ne11, int64_t dst_s1, cudaStream_t stream);
