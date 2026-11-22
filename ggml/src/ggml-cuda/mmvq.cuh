#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
#define MMVQ_GCN_WIDE_MAX_BATCH 16 // GCN: Q4_0/Q4_1/Q8_0 up to this batch size.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11, int64_t nrows_x);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

// A q8_1 buffer for node node_idx of the graph being evaluated that later mat-vecs on it will use, if one of them
// follows; the caller fills it in the layout of quantize_q8_1 or resets ctx.q8_1_cache.
char * ggml_cuda_q8_1_cache_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int node_idx);

// gfx906: Q8_0 weights are rearranged into planes when their last bytes are uploaded; the mat-vec and MMQ read them.
bool ggml_cuda_q8_0_planes_eligible(const ggml_tensor * tensor, int device);
void ggml_cuda_q8_0_planes_repack(const ggml_tensor * tensor, cudaStream_t stream);
bool ggml_cuda_q8_0_planes_get(const ggml_tensor * tensor, void * data, size_t offset, size_t size, cudaStream_t stream);
bool ggml_cuda_q8_0_planes_set(const ggml_tensor * tensor, const void * data, size_t offset, size_t size, cudaStream_t stream);
void ggml_cuda_q8_0_planes_forget(const void * base, size_t size);
bool ggml_cuda_q8_0_planes(const ggml_tensor * tensor);
// Writes the canonical layout of the tensor's bytes, rounded up to whole groups of 8 blocks, to dst.
void ggml_cuda_q8_0_planes_unpack(const ggml_tensor * tensor, void * dst, cudaStream_t stream);
// gfx906: Q6_K weights are rearranged into planes when their last bytes are uploaded; only the mat-vec and MMQ read them then.
bool ggml_cuda_q6_k_planes_eligible(const ggml_tensor * tensor, int device);
void ggml_cuda_q6_k_planes_repack(const ggml_tensor * tensor, cudaStream_t stream);
bool ggml_cuda_q6_k_planes_get(const ggml_tensor * tensor, void * data, size_t offset, size_t size, cudaStream_t stream);
bool ggml_cuda_q6_k_planes_set(const ggml_tensor * tensor, const void * data, size_t offset, size_t size, cudaStream_t stream);
void ggml_cuda_q6_k_planes_forget(const void * base, size_t size);
int64_t ggml_cuda_q6_k_planes_ne0(const ggml_tensor * src0);
bool ggml_cuda_q6_k_planes(const ggml_tensor * src0);
// The whole tensor in planes that src0 points into, in the canonical layout: its size, and where src0's data lands in dst.
size_t ggml_cuda_q6_k_planes_nbytes(const ggml_tensor * src0);
char * ggml_cuda_q6_k_planes_unpack(const ggml_tensor * src0, char * dst, cudaStream_t stream);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

bool ggml_cuda_should_use_mmvq_gdn_gate(const ggml_tensor * mm_a, const ggml_tensor * mm_b, int cc);

// The Qwen3.5/3.8 GDN gate (see ggml_cuda_try_fuse_gdn_gate) with quantized alpha/beta weights in one launch.
void ggml_cuda_mul_mat_vec_q_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * mm_b,
    const ggml_tensor * dt, const ggml_tensor * a, ggml_tensor * dst_a, ggml_tensor * dst_b);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
