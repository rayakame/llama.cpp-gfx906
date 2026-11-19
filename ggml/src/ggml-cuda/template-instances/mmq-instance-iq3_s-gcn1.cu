// The gfx906 configs of mmq-config-gcn.cuh (GCN config variant 1).

#define GGML_MMQ_GCN_VARIANT 1
#include "../mmq.cuh"

DECL_MMQ_CASE_GCN_VARIANT(GGML_TYPE_IQ3_S, 1);
