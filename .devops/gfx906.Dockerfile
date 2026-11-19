# llama.cpp for the AMD Instinct MI50 (gfx906), built from this tree on a ROCm 10.1 image that still ships gfx906
# kernels for rocBLAS.
ARG ROCM_IMAGE=docker.io/mixa3607/rocm-gfx906:10.1-complete-20261018172427

FROM ${ROCM_IMAGE} AS build
# The upstream tag this branch is based on; it also names the prebuilt web UI to download.
ARG UPSTREAM_REF=b12230
# ON adds test-backend-ops and the other tests to /app.
ARG TESTS=OFF
# OFF builds without the web UI, for a tree whose UI version has no prebuilt download.
ARG PREBUILT_UI=ON
# The image's own gfx906 apt source is sometimes down; the build tools come from Ubuntu's.
RUN (apt-get update || true) && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build libssl-dev git && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN export HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" HF_UI_VERSION="${UPSTREAM_REF}" && \
    cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS="${TESTS}" -DLLAMA_USE_PREBUILT_UI="${PREBUILT_UI}" \
      -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
      -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON -DGGML_RPC=ON \
      -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON -DAMDGPU_TARGETS=gfx906 \
      "-DCMAKE_HIP_FLAGS=-mllvm -amdgpu-sched-strategy=max-ilp" && \
    cmake --build /build -j"$(nproc)"

FROM ${ROCM_IMAGE}
RUN (apt-get update || true) && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libgomp1 libssl3 curl && rm -rf /var/lib/apt/lists/*
COPY --from=build /build/bin/ /app/
WORKDIR /app
RUN cd / && ! ldd /app/llama-server | grep "not found"
ENTRYPOINT ["/app/llama-server"]
