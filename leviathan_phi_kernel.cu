// leviathan_phi_kernel.cu
// GPU-Accelerated IIT Φ Solver — Native CUDA replacement for PyPhi
// Computes Integrated Information (Φ) entirely on GPU
// Supports up to 10 binary hub nodes (1024 states, 511 bipartitions)
// v3.3 Optimized: shared memory, warp reduction, zero per-call alloc

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define PHI_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "PHI CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// GPU Φ Handle
// ---------------------------------------------------------------------------

struct PhiHandle {
  int num_hubs;
  int num_states;     // 2^num_hubs for binary
  int num_partitions; // 2^(num_hubs-1) - 1 bipartitions

  int *d_hub_indices; // [num_hubs] — global node indices of hubs
  int *d_prev_state;  // [1] — previous binarized state
  int *d_state_out;   // [OPT #5] Pre-allocated (was per-call cudaMalloc)

  // TPM accumulation (on GPU, no CPU roundtrip)
  int64_t *d_tpm_counts; // [num_states * num_states] — transition counts
  int64_t *d_row_totals; // [num_states] — total transitions from each state
  int64_t transition_counter;

  // Normalized TPM (float)
  float *d_tpm;     // [num_states * num_states] — P(t+1|t)
  float *d_sbn_tpm; // [num_states * num_hubs]   — state-by-node TPM

  // Φ computation workspace
  float *d_phi_per_partition; // [num_partitions] — φ for each bipartition

  // Result (pinned for async transfer)
  float *d_phi_result; // [1] — final Φ on device
  float *h_phi_pinned; // [1] — pinned host memory

  // CUDA resources
  cudaStream_t stream;
};

// ---------------------------------------------------------------------------
// Kernel 1: Binarize & Accumulate TPM (called every physics step)
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(1, 1) void binarize_and_accumulate_kernel(
    const int num_hubs, const int num_states, const float *__restrict__ d_theta,
    const int *__restrict__ d_hub_indices, int *__restrict__ d_prev_state,
    int64_t *__restrict__ d_tpm_counts, // [OPT #22]
    int64_t *__restrict__ d_row_totals, // [OPT #22]
    int *__restrict__ d_state_out) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  // Binarize: phase > π → 1, else → 0
  int state = 0;
  for (int i = 0; i < num_hubs; i++) {
    int node_idx = __ldg(&d_hub_indices[i]);
    float phase = d_theta[node_idx];
    phase = fmodf(phase, 6.2831853f);
    if (phase < 0)
      phase += 6.2831853f;
    if (phase > 3.1415926f) {
      state |= (1 << i);
    }
  }

  int prev = *d_prev_state;

  if (prev >= 0 && prev < num_states) {
    atomicAdd((unsigned long long *)&d_tpm_counts[prev * num_states + state],
              1ULL);
    atomicAdd((unsigned long long *)&d_row_totals[prev], 1ULL);
  }

  *d_prev_state = state;
  *d_state_out = state;
}

// ---------------------------------------------------------------------------
// Kernel 2: Normalize TPM (transition counts → probabilities)
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4) void normalize_tpm_kernel(
    const int num_states, const int64_t *__restrict__ d_tpm_counts,
    const int64_t *__restrict__ d_row_totals,
    float *__restrict__ d_tpm // [OPT #22]
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_states * num_states;
  if (idx >= total)
    return;

  int source = idx / num_states;
  int64_t row_total = d_row_totals[source];

  d_tpm[idx] = (row_total > 0) ? __fdividef((float)d_tpm_counts[idx],
                                            (float)row_total) // Fast divide
                               : __fdividef(1.0f, (float)num_states);
}

// ---------------------------------------------------------------------------
// Kernel 3: Build State-by-Node TPM
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4) void build_sbn_tpm_kernel(
    const int num_states, const int num_hubs, const float *__restrict__ d_tpm,
    float *__restrict__ d_sbn_tpm // [OPT #22]
) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_states)
    return;

  for (int n = 0; n < num_hubs; n++) {
    float prob_on = 0.0f;
    for (int t = 0; t < num_states; t++) {
      // [OPT] Branchless bit extraction
      prob_on += d_tpm[s * num_states + t] * (float)((t >> n) & 1);
    }
    d_sbn_tpm[s * num_hubs + n] = prob_on;
  }
}

// ---------------------------------------------------------------------------
// Kernel 4: Compute φ for each bipartition
// [OPT #6] Shared memory for SBN-TPM, [OPT #12] __popc for popcount
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(256, 2) void compute_phi_partitions_kernel(
    const int num_states, const int num_hubs, const int num_partitions,
    const float *__restrict__ d_sbn_tpm,
    float *__restrict__ d_phi_per_partition // [OPT #22]
) {
  // [OPT #6] Load entire SBN-TPM into shared memory
  // Max: 10 hubs × 1024 states × 4B = 40KB (fits in H100's 228KB smem)
  extern __shared__ float s_sbn_tpm[];

  // Cooperative loading
  int total_sbn = num_states * num_hubs;
  for (int i = threadIdx.x; i < total_sbn; i += blockDim.x) {
    s_sbn_tpm[i] = d_sbn_tpm[i];
  }
  __syncthreads();

  int part_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_idx >= num_partitions)
    return;

  int mask = part_idx + 1;
  int hub_mask = (1 << num_hubs) - 1;

  // [OPT #12] Hardware popcount
  int n_A = __popc(mask & hub_mask);
  int n_B = num_hubs - n_A;

  if (n_A == 0 || n_B == 0 || n_A == num_hubs) {
    d_phi_per_partition[part_idx] = 1e30f;
    return;
  }

  float phi = 0.0f;

  for (int s = 0; s < num_states; s++) {
    float p_integrated = 1.0f;
    float p_partA = 1.0f;
    float p_partB = 1.0f;

    for (int n = 0; n < num_hubs; n++) {
      // [OPT #6] Read from shared memory instead of global
      float p_n = s_sbn_tpm[s * num_hubs + n];
      int node_state = (s >> n) & 1;
      float prob = node_state ? p_n : (1.0f - p_n);
      prob = fmaxf(prob, 1e-10f);

      p_integrated *= prob;

      if (mask & (1 << n)) {
        p_partA *= prob;
      } else {
        p_partB *= prob;
      }
    }

    float p_partitioned = p_partA * p_partB;

    // Fast log using intrinsic
    float log_int = __logf(fmaxf(p_integrated, 1e-30f));
    float log_part = __logf(fmaxf(p_partitioned, 1e-30f));
    phi += fabsf(log_int - log_part);
  }

  d_phi_per_partition[part_idx] = phi * __fdividef(1.0f, (float)num_states);
}

// ---------------------------------------------------------------------------
// Kernel 5: Find MIP — [OPT #7] Warp-parallel reduction
// ---------------------------------------------------------------------------

__global__ void find_mip_kernel(const int num_partitions,
                                const float *__restrict__ d_phi_per_partition,
                                float *__restrict__ d_phi_result // [OPT #22]
) {
  // Single warp: each lane handles a stride of partitions
  float min_phi = 1e30f;
  int lane = threadIdx.x;

  for (int i = lane; i < num_partitions; i += 32) {
    min_phi = fminf(min_phi, d_phi_per_partition[i]);
  }

  // [OPT #7] Warp-level min reduction
  unsigned int mask = 0xffffffff;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    min_phi = fminf(min_phi, __shfl_down_sync(mask, min_phi, offset));
  }

  if (lane == 0) {
    *d_phi_result = min_phi;
  }
}

// ---------------------------------------------------------------------------
// Host API (extern "C" for Pybind11)
// ---------------------------------------------------------------------------

extern "C" {

PhiHandle *phi_init(int num_hubs, int *h_hub_indices) {
  if (num_hubs > 10) {
    fprintf(stderr, "PHI ERROR: max 10 hubs supported (got %d)\n", num_hubs);
    return nullptr;
  }

  PhiHandle *h = new PhiHandle();
  h->num_hubs = num_hubs;
  h->num_states = 1 << num_hubs;
  h->num_partitions = (1 << (num_hubs - 1)) - 1;
  h->transition_counter = 0;

  int S = h->num_states;
  int P = h->num_partitions;

  // Allocate device memory
  PHI_CHECK(cudaMalloc(&h->d_hub_indices, num_hubs * sizeof(int)));
  PHI_CHECK(cudaMalloc(&h->d_prev_state, sizeof(int)));
  PHI_CHECK(cudaMalloc(&h->d_state_out, sizeof(int))); // [OPT #5]
  PHI_CHECK(cudaMalloc(&h->d_tpm_counts, S * S * sizeof(int64_t)));
  PHI_CHECK(cudaMalloc(&h->d_row_totals, S * sizeof(int64_t)));
  PHI_CHECK(cudaMalloc(&h->d_tpm, S * S * sizeof(float)));
  PHI_CHECK(cudaMalloc(&h->d_sbn_tpm, S * num_hubs * sizeof(float)));
  PHI_CHECK(cudaMalloc(&h->d_phi_per_partition, P * sizeof(float)));
  PHI_CHECK(cudaMalloc(&h->d_phi_result, sizeof(float)));

  // [OPT] Pinned host memory for result
  PHI_CHECK(cudaMallocHost(&h->h_phi_pinned, sizeof(float)));

  // Initialize
  PHI_CHECK(cudaMemcpy(h->d_hub_indices, h_hub_indices, num_hubs * sizeof(int),
                       cudaMemcpyHostToDevice));
  int neg_one = -1;
  PHI_CHECK(cudaMemcpy(h->d_prev_state, &neg_one, sizeof(int),
                       cudaMemcpyHostToDevice));
  PHI_CHECK(cudaMemset(h->d_tpm_counts, 0, S * S * sizeof(int64_t)));
  PHI_CHECK(cudaMemset(h->d_row_totals, 0, S * sizeof(int64_t)));

  PHI_CHECK(cudaStreamCreate(&h->stream));

  printf("[PhiGPU] Initialized: %d hubs, %d states, %d partitions\n", num_hubs,
         S, P);

  return h;
}

void phi_accumulate(PhiHandle *handle, const float *d_theta) {
  // [OPT #5] No per-call cudaMalloc — use pre-allocated d_state_out
  binarize_and_accumulate_kernel<<<1, 1, 0, handle->stream>>>(
      handle->num_hubs, handle->num_states, d_theta, handle->d_hub_indices,
      handle->d_prev_state, handle->d_tpm_counts, handle->d_row_totals,
      handle->d_state_out);

  // [OPT #11] No sync — accumulate is fire-and-forget
  handle->transition_counter++;
}

float phi_compute(PhiHandle *handle) {
  int S = handle->num_states;
  int P = handle->num_partitions;
  int H = handle->num_hubs;

  // 1. Normalize TPM
  int tpm_threads = 256;
  int tpm_blocks = (S * S + tpm_threads - 1) / tpm_threads;
  normalize_tpm_kernel<<<tpm_blocks, tpm_threads, 0, handle->stream>>>(
      S, handle->d_tpm_counts, handle->d_row_totals, handle->d_tpm);

  // 2. Build state-by-node TPM
  int sbn_blocks = (S + 255) / 256;
  build_sbn_tpm_kernel<<<sbn_blocks, 256, 0, handle->stream>>>(
      S, H, handle->d_tpm, handle->d_sbn_tpm);

  // 3. Compute φ for each bipartition
  // [OPT #6] Dynamic shared memory for SBN-TPM
  size_t smem_bytes = S * H * sizeof(float);
  int phi_blocks = (P + 255) / 256;
  compute_phi_partitions_kernel<<<phi_blocks, 256, smem_bytes,
                                  handle->stream>>>(
      S, H, P, handle->d_sbn_tpm, handle->d_phi_per_partition);

  // 4. Find MIP — [OPT #7] warp-parallel
  find_mip_kernel<<<1, 32, 0, handle->stream>>>(P, handle->d_phi_per_partition,
                                                handle->d_phi_result);

  // 5. Async copy to pinned host
  PHI_CHECK(cudaMemcpyAsync(handle->h_phi_pinned, handle->d_phi_result,
                            sizeof(float), cudaMemcpyDeviceToHost,
                            handle->stream));
  PHI_CHECK(cudaStreamSynchronize(handle->stream));

  return *(handle->h_phi_pinned);
}

void phi_reset(PhiHandle *handle) {
  int S = handle->num_states;
  PHI_CHECK(cudaMemset(handle->d_tpm_counts, 0, S * S * sizeof(int64_t)));
  PHI_CHECK(cudaMemset(handle->d_row_totals, 0, S * sizeof(int64_t)));
  int neg_one = -1;
  PHI_CHECK(cudaMemcpy(handle->d_prev_state, &neg_one, sizeof(int),
                       cudaMemcpyHostToDevice));
  handle->transition_counter = 0;
}

int64_t phi_get_transition_count(PhiHandle *handle) {
  return handle->transition_counter;
}

void phi_free(PhiHandle *handle) {
  if (!handle)
    return;
  cudaFree(handle->d_hub_indices);
  cudaFree(handle->d_prev_state);
  cudaFree(handle->d_state_out);
  cudaFree(handle->d_tpm_counts);
  cudaFree(handle->d_row_totals);
  cudaFree(handle->d_tpm);
  cudaFree(handle->d_sbn_tpm);
  cudaFree(handle->d_phi_per_partition);
  cudaFree(handle->d_phi_result);
  cudaFreeHost(handle->h_phi_pinned);
  cudaStreamDestroy(handle->stream);
  delete handle;
  printf("[PhiGPU] Resources freed\n");
}

} // extern "C"
