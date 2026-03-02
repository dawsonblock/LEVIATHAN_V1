// leviathan_kernel.cu
// GPU-accelerated delayed Kuramoto network with adaptive coupling
// Targets SM86 (3080 Ti) / SM90 (H100)

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <math_constants.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

extern "C" {

// ============================================================================
// 1. DATA STRUCTURES & HYPERPARAMETERS
// ============================================================================

struct LeviathanHandle {
  int N;
  int max_delay;
  int buffer_size;
  int num_edges;
  int buf_idx;

  int blocks;
  int threads;

  // Physics parameters
  float eta, lam, gamma, alpha_eps, kappa_w;
  float w_min, w_max;
  float g, r0, beta, mu;
  float g_min, g_max;

  // Device topology arrays (read-only after init)
  int *d_row_ptr;
  int *d_col_idx;
  uint8_t *d_delays;
  float *d_weights;

  // Device state arrays
  float *d_theta;
  float *d_theta_hat;
  float *d_omega;
  float *d_omega_hat;
  float *d_eps;
  float *d_sum_coupling;
  float *d_theta_buffer;

  // Reduction buffers for Macroscopic Order Parameter (r)
  float *d_block_cos;
  float *d_block_sin;
  float *h_block_cos; // [OPT #1] Pinned memory
  float *h_block_sin; // [OPT #1] Pinned memory

  // [OPT #2] GPU-side final reduction result
  float *d_r_result; // [1] — final r on device
  float *h_r_pinned; // [1] — pinned host for async D→H

  cudaStream_t stream;
};

// ============================================================================
// 2. DEVICE KERNELS (all with __launch_bounds__ for SM90 occupancy)
// ============================================================================

// Initializes the circular history buffer to prevent cold-start shocks
__global__ __launch_bounds__(256, 4) // [OPT #4]
    void init_buffer_kernel(const int N, const int buffer_size,
                            const float *__restrict__ theta,
                            float *__restrict__ theta_buffer) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float val = theta[i];
    for (int idx = 0; idx < buffer_size; ++idx) {
      theta_buffer[i * buffer_size + idx] = val;
    }
  }
}

// Computes prediction error and advances internal active-inference models
__global__ __launch_bounds__(256, 4) // [OPT #4]
    void prep_kernel(const int N, const float *__restrict__ theta,
                     float *__restrict__ theta_hat,
                     float *__restrict__ omega_hat, float *__restrict__ eps,
                     const float dt, // [OPT #21] const
                     const float kappa_w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float diff = theta[i] - theta_hat[i];
    const float TWO_PI = 2.0f * CUDART_PI_F;

    // Fast symmetric wrap to [-pi, pi]
    diff = diff - floorf(diff / TWO_PI + 0.5f) * TWO_PI;
    eps[i] = sinf(diff);

    theta_hat[i] += dt * omega_hat[i];
    omega_hat[i] += kappa_w * dt * eps[i];
  }
}

// Node-centric CSR physics solver. ZERO ATOMICS. Thread i owns node i.
__global__ __launch_bounds__(256, 4) // [OPT #4]
    void csr_physics_kernel(
        const int N, const int *__restrict__ row_ptr,
        const int *__restrict__ col_idx, const uint8_t *__restrict__ delays,
        float *__restrict__ weights, const float *__restrict__ theta,
        const float *__restrict__ theta_buffer, const int buf_idx,
        const int buffer_size, // [OPT #21] const
        const float *__restrict__ eps, const float dt, const float eta,
        const float gamma, const float lam, const float w_min,
        const float w_max, float *__restrict__ sum_coupling) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float local_sum = 0.0f;
    float my_eps = fabsf(eps[i]);
    float my_theta = theta[i];
    const float TWO_PI = 2.0f * CUDART_PI_F;

    // [OPT #9] Use __ldg for read-only topology arrays
    int start = __ldg(&row_ptr[i]);
    int end = __ldg(&row_ptr[i + 1]);

// Loop strictly over incoming edges to Node i
#pragma unroll 8 // [OPT #20] Typical degree ~20
    for (int e = start; e < end; ++e) {
      int j = __ldg(&col_idx[e]); // [OPT #9]
      int d = __ldg(&delays[e]);  // [OPT #9]

      // L2-Cached delayed phase gather
      int idx = (buf_idx - d + buffer_size) % buffer_size;
      float delayed_j = theta_buffer[j * buffer_size + idx];

      float diff = delayed_j - my_theta;
      diff = diff - floorf(diff / TWO_PI + 0.5f) * TWO_PI;

      float w = weights[e];

      // [OPT #3] Fuse sin+cos into single __sincosf call
      float sin_diff, cos_diff;
      __sincosf(diff, &sin_diff, &cos_diff);

      local_sum += w * sin_diff;

      // Precision-weighted Hebbian learning
      float dw = dt * (eta * cos_diff * (1.0f - gamma * my_eps) - lam * w);

      w += dw;
      if (!isfinite(w))
        w = w_min; // NaN guard
      w = fmaxf(w_min, fminf(w_max, w));
      weights[e] = w;
    }
    sum_coupling[i] = local_sum;
  }
}

// Integrates phase, writes to buffer, and computes in-VRAM warp reductions
__global__ __launch_bounds__(256, 4) // [OPT #4]
    void integrate_and_reduce_kernel(
        const int N, float *__restrict__ theta, const float *__restrict__ omega,
        const float *__restrict__ sum_coupling, const float g,
        const float *__restrict__ eps, const float dt, const float alpha_eps,
        float *__restrict__ theta_buffer, const int next_buf_idx,
        const int buffer_size, // [OPT #21]
        float *__restrict__ block_cos, float *__restrict__ block_sin) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float my_cos = 0.0f;
  float my_sin = 0.0f;

  if (i < N) {
    float inc = dt * (omega[i] + g * sum_coupling[i] - alpha_eps * eps[i]);
    float x = theta[i] + inc;
    const float TWO_PI = 2.0f * CUDART_PI_F;

    // Fast Wrap to [0, 2pi)
    x = x - floorf(x / TWO_PI) * TWO_PI;
    if (!isfinite(x))
      x = 0.0f; // NaN guard
    theta[i] = x;
    theta_buffer[i * buffer_size + next_buf_idx] = x;

    // [OPT #3] Fuse sin+cos
    __sincosf(x, &my_sin, &my_cos);
  }

  // Hopper-safe unconditional warp reduction
  unsigned int mask = 0xffffffff;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    my_cos += __shfl_down_sync(mask, my_cos, offset);
    my_sin += __shfl_down_sync(mask, my_sin, offset);
  }

  __shared__ float sm_cos[32];
  __shared__ float sm_sin[32];

  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

  if (lane == 0) {
    sm_cos[warp] = my_cos;
    sm_sin[warp] = my_sin;
  }
  __syncthreads();

  // Final warp reduces the shared memory block sums
  if (warp == 0) {
    my_cos = (lane < (blockDim.x / 32)) ? sm_cos[lane] : 0.0f;
    my_sin = (lane < (blockDim.x / 32)) ? sm_sin[lane] : 0.0f;

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      my_cos += __shfl_down_sync(mask, my_cos, offset);
      my_sin += __shfl_down_sync(mask, my_sin, offset);
    }

    if (lane == 0) {
      block_cos[blockIdx.x] = my_cos;
      block_sin[blockIdx.x] = my_sin;
    }
  }
}

// [OPT #2] GPU-side final reduction: block sums → single (r) value
__global__ void final_reduce_r_kernel(const int num_blocks, const int N,
                                      const float *__restrict__ block_cos,
                                      const float *__restrict__ block_sin,
                                      float *__restrict__ r_result) {
  // Single warp handles up to 1024 blocks
  float sum_cos = 0.0f;
  float sum_sin = 0.0f;
  int lane = threadIdx.x;

  // Stride loop: each lane accumulates every 32nd block
  for (int i = lane; i < num_blocks; i += 32) {
    sum_cos += block_cos[i];
    sum_sin += block_sin[i];
  }

  // Warp reduce
  unsigned int mask = 0xffffffff;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum_cos += __shfl_down_sync(mask, sum_cos, offset);
    sum_sin += __shfl_down_sync(mask, sum_sin, offset);
  }

  if (lane == 0) {
    float inv_N = 1.0f / (float)N;
    r_result[0] = sqrtf(sum_cos * sum_cos + sum_sin * sum_sin) * inv_N;
  }
}

// ============================================================================
// 3. HOST API (Bound to Python via Pybind11)
// ============================================================================

LeviathanHandle *leviathan_init(int N, int max_delay, int *h_row_ptr,
                                int *h_col_idx, uint8_t *h_delays,
                                float *h_weights, float *h_theta,
                                float *h_theta_hat, float *h_omega) {
  LeviathanHandle *handle = new LeviathanHandle();
  handle->N = N;
  handle->max_delay = max_delay;
  handle->buffer_size = max_delay + 1;
  handle->num_edges = h_row_ptr[N];
  handle->buf_idx = 0;

  // Scientific Parameters
  handle->eta = 0.01f;
  handle->lam = 0.001f;
  handle->gamma = 0.05f;
  handle->alpha_eps = 0.1f;
  handle->kappa_w = 0.01f;
  handle->w_min = 0.0f;
  handle->w_max = 2.0f;

  handle->g = 1.5f;
  handle->r0 = 0.5f;
  handle->beta = 0.01f;
  handle->mu = 0.01f;
  handle->g_min = 0.1f;
  handle->g_max = 5.0f;

  CUDA_CHECK(cudaStreamCreate(&handle->stream));

  // Allocate Device Memory
  CUDA_CHECK(cudaMalloc(&handle->d_row_ptr, (N + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&handle->d_col_idx, handle->num_edges * sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(&handle->d_delays, handle->num_edges * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&handle->d_weights, handle->num_edges * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&handle->d_theta, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_theta_hat, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_omega, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_omega_hat, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_eps, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_sum_coupling, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_theta_buffer,
                        N * handle->buffer_size * sizeof(float)));

  // Host to Device Transfers
  CUDA_CHECK(cudaMemcpyAsync(handle->d_row_ptr, h_row_ptr,
                             (N + 1) * sizeof(int), cudaMemcpyHostToDevice,
                             handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_col_idx, h_col_idx,
                             handle->num_edges * sizeof(int),
                             cudaMemcpyHostToDevice, handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_delays, h_delays,
                             handle->num_edges * sizeof(uint8_t),
                             cudaMemcpyHostToDevice, handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_weights, h_weights,
                             handle->num_edges * sizeof(float),
                             cudaMemcpyHostToDevice, handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_theta, h_theta, N * sizeof(float),
                             cudaMemcpyHostToDevice, handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_theta_hat, h_theta_hat,
                             N * sizeof(float), cudaMemcpyHostToDevice,
                             handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_omega, h_omega, N * sizeof(float),
                             cudaMemcpyHostToDevice, handle->stream));
  CUDA_CHECK(cudaMemcpyAsync(handle->d_omega_hat, h_omega, N * sizeof(float),
                             cudaMemcpyHostToDevice, handle->stream));

  // Launch Configuration
  handle->threads = 256;
  handle->blocks = (N + handle->threads - 1) / handle->threads;

  CUDA_CHECK(cudaMalloc(&handle->d_block_cos, handle->blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&handle->d_block_sin, handle->blocks * sizeof(float)));

  // [OPT #1] Pinned memory for async D→H transfers
  CUDA_CHECK(
      cudaMallocHost(&handle->h_block_cos, handle->blocks * sizeof(float)));
  CUDA_CHECK(
      cudaMallocHost(&handle->h_block_sin, handle->blocks * sizeof(float)));

  // [OPT #2] GPU-side final reduction buffers
  CUDA_CHECK(cudaMalloc(&handle->d_r_result, sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&handle->h_r_pinned, sizeof(float)));

  // Cold-start Buffer Population
  init_buffer_kernel<<<handle->blocks, handle->threads, 0, handle->stream>>>(
      N, handle->buffer_size, handle->d_theta, handle->d_theta_buffer);

  // H100 SM90 L2 Cache Persistence
#if __CUDACC_VER_MAJOR__ >= 12
  cudaStreamAttrValue attr;
  attr.accessPolicyWindow.base_ptr =
      reinterpret_cast<void *>(handle->d_theta_buffer);
  attr.accessPolicyWindow.num_bytes = N * handle->buffer_size * sizeof(float);
  attr.accessPolicyWindow.hitRatio = 1.0;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cudaStreamSetAttribute(handle->stream, cudaStreamAttributeAccessPolicyWindow,
                         &attr);
#endif

  CUDA_CHECK(cudaStreamSynchronize(handle->stream));
  return handle;
}

float leviathan_step(LeviathanHandle *handle, float dt) {
  // 1. Internal Predictor
  prep_kernel<<<handle->blocks, handle->threads, 0, handle->stream>>>(
      handle->N, handle->d_theta, handle->d_theta_hat, handle->d_omega_hat,
      handle->d_eps, dt, handle->kappa_w);

  // 2. Physics & Plasticity (Zero Atomics CSR)
  csr_physics_kernel<<<handle->blocks, handle->threads, 0, handle->stream>>>(
      handle->N, handle->d_row_ptr, handle->d_col_idx, handle->d_delays,
      handle->d_weights, handle->d_theta, handle->d_theta_buffer,
      handle->buf_idx, handle->buffer_size, handle->d_eps, dt, handle->eta,
      handle->gamma, handle->lam, handle->w_min, handle->w_max,
      handle->d_sum_coupling);

  // 3. Integration & Macroscopic Reduction
  int next_buf_idx = (handle->buf_idx + 1) % handle->buffer_size;

  integrate_and_reduce_kernel<<<handle->blocks, handle->threads, 0,
                                handle->stream>>>(
      handle->N, handle->d_theta, handle->d_omega, handle->d_sum_coupling,
      handle->g, handle->d_eps, dt, handle->alpha_eps, handle->d_theta_buffer,
      next_buf_idx, handle->buffer_size, handle->d_block_cos,
      handle->d_block_sin);

  handle->buf_idx = next_buf_idx;

  // [OPT #2] GPU-side final reduction — single warp reduces all block sums
  final_reduce_r_kernel<<<1, 32, 0, handle->stream>>>(
      handle->blocks, handle->N, handle->d_block_cos, handle->d_block_sin,
      handle->d_r_result);

  // [OPT #1 + #10] Single float async D→H with pinned memory
  CUDA_CHECK(cudaMemcpyAsync(handle->h_r_pinned, handle->d_r_result,
                             sizeof(float), cudaMemcpyDeviceToHost,
                             handle->stream));
  CUDA_CHECK(cudaStreamSynchronize(handle->stream));

  float r = *(handle->h_r_pinned);

  // 5. Metastability Gain Controller
  handle->g +=
      dt * (handle->beta * (handle->r0 - r) - handle->mu * (handle->g - 1.5f));
  handle->g = std::fmax(handle->g_min, std::fmin(handle->g_max, handle->g));

  return r;
}

void leviathan_get_theta(LeviathanHandle *handle, float *h_theta) {
  CUDA_CHECK(cudaMemcpy(h_theta, handle->d_theta, handle->N * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void leviathan_set_theta(LeviathanHandle *handle, float *h_theta) {
  CUDA_CHECK(cudaMemcpyAsync(handle->d_theta, h_theta,
                             handle->N * sizeof(float), cudaMemcpyHostToDevice,
                             handle->stream));
  // Update history buffer with new phases to prevent discontinuity
  init_buffer_kernel<<<handle->blocks, handle->threads, 0, handle->stream>>>(
      handle->N, handle->buffer_size, handle->d_theta, handle->d_theta_buffer);
  CUDA_CHECK(cudaStreamSynchronize(handle->stream));
}

void leviathan_free(LeviathanHandle *handle) {
  cudaFree(handle->d_row_ptr);
  cudaFree(handle->d_col_idx);
  cudaFree(handle->d_delays);
  cudaFree(handle->d_weights);
  cudaFree(handle->d_theta);
  cudaFree(handle->d_theta_hat);
  cudaFree(handle->d_omega);
  cudaFree(handle->d_omega_hat);
  cudaFree(handle->d_eps);
  cudaFree(handle->d_sum_coupling);
  cudaFree(handle->d_theta_buffer);
  cudaFree(handle->d_block_cos);
  cudaFree(handle->d_block_sin);
  cudaFree(handle->d_r_result);

  // [OPT #1] Free pinned memory
  cudaFreeHost(handle->h_block_cos);
  cudaFreeHost(handle->h_block_sin);
  cudaFreeHost(handle->h_r_pinned);

  cudaStreamDestroy(handle->stream);
  delete handle;
}

} // extern "C"