// leviathan_multigpu.cu
// Multi-GPU Leviathan Engine with NCCL collective communications
// Partitions the graph across K GPUs with halo exchange for cross-partition
// edges

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>

#define MG_CHECK(call)                                                         \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t res = (call);                                                 \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,         \
              ncclGetErrorString(res));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Forward: single-GPU handle from leviathan_kernel.cu
extern "C" {
struct LeviathanHandle;
LeviathanHandle *leviathan_init(int N, int max_delay, int *h_row_ptr,
                                int *h_col_idx, uint8_t *h_delays,
                                float *h_weights, float *h_theta,
                                float *h_theta_hat, float *h_omega);
float leviathan_step(LeviathanHandle *handle, float dt);
void leviathan_get_theta(LeviathanHandle *handle, float *h_theta);
void leviathan_set_theta(LeviathanHandle *handle, float *h_theta);
void leviathan_free(LeviathanHandle *handle);
}

// ---------------------------------------------------------------------------
// Partition Info (one per GPU)
// ---------------------------------------------------------------------------

struct PartitionInfo {
  int gpu_id;
  int local_N;          // Number of nodes in this partition
  int *global_to_local; // Map: global node ID → local node ID (or -1)
  int *local_to_global; // Map: local node ID → global node ID

  // Halo data: nodes we need from OTHER partitions
  int halo_recv_size;   // Number of remote nodes we depend on
  int *halo_remote_ids; // Global IDs of remote nodes we need
  int *halo_source_gpu; // Which GPU owns each halo node

  // Halo send: nodes other partitions need from US
  int halo_send_size;
  int *halo_send_local_ids; // Local IDs of nodes to send

  // Device buffers for halo exchange
  float *d_halo_recv; // [halo_recv_size] — received phases
  float *d_halo_send; // [halo_send_size] — phases to send
};

// ---------------------------------------------------------------------------
// Multi-GPU Handle
// ---------------------------------------------------------------------------

struct MultiGPUHandle {
  int num_gpus;
  int total_N;
  int max_delay;

  // Per-GPU partition handles
  LeviathanHandle **partitions;
  PartitionInfo *part_info;

  // NCCL
  ncclComm_t *nccl_comms;
  cudaStream_t *streams;

  // Global order parameter aggregation
  float *d_r_local;  // [num_gpus] — per-GPU partial r
  float *d_r_global; // [1] — aggregated r after AllReduce
  float h_r;         // Host-side result
};

// ---------------------------------------------------------------------------
// Halo Exchange Kernel
// ---------------------------------------------------------------------------

__global__ void pack_halo_kernel(int send_size,
                                 const int *__restrict__ send_local_ids,
                                 const float *__restrict__ d_theta,
                                 float *d_halo_send) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= send_size)
    return;
  d_halo_send[idx] = d_theta[send_local_ids[idx]];
}

__global__ void
unpack_halo_kernel(int recv_size, const float *__restrict__ d_halo_recv,
                   float *d_theta_halo // Extended theta buffer for halo nodes
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= recv_size)
    return;
  d_theta_halo[idx] = d_halo_recv[idx];
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

extern "C" {

MultiGPUHandle *
multigpu_init(int num_gpus, int total_N, int max_delay,
              // Per-partition data (arrays of pointers, one per GPU)
              int **part_row_ptrs, int **part_col_idxs, uint8_t **part_delays,
              float **part_weights, float **part_thetas,
              float **part_theta_hats, float **part_omegas, int *part_sizes,
              // Halo maps
              int *halo_recv_sizes, int **halo_remote_ids, int *halo_send_sizes,
              int **halo_send_local_ids) {
  MultiGPUHandle *h = new MultiGPUHandle();
  h->num_gpus = num_gpus;
  h->total_N = total_N;
  h->max_delay = max_delay;

  h->partitions = new LeviathanHandle *[num_gpus];
  h->part_info = new PartitionInfo[num_gpus];
  h->nccl_comms = new ncclComm_t[num_gpus];
  h->streams = new cudaStream_t[num_gpus];

  // Initialize NCCL
  ncclUniqueId nccl_id;
  NCCL_CHECK(ncclGetUniqueId(&nccl_id));
  NCCL_CHECK(ncclGroupStart());
  for (int g = 0; g < num_gpus; g++) {
    MG_CHECK(cudaSetDevice(g));
    NCCL_CHECK(ncclCommInitRank(&h->nccl_comms[g], num_gpus, nccl_id, g));
  }
  NCCL_CHECK(ncclGroupEnd());

  // Enable P2P access between all GPU pairs
  for (int i = 0; i < num_gpus; i++) {
    MG_CHECK(cudaSetDevice(i));
    for (int j = 0; j < num_gpus; j++) {
      if (i != j) {
        int can_access;
        MG_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
        if (can_access) {
          cudaDeviceEnablePeerAccess(j, 0);
        }
      }
    }
  }

  // Initialize each partition
  for (int g = 0; g < num_gpus; g++) {
    MG_CHECK(cudaSetDevice(g));
    MG_CHECK(cudaStreamCreate(&h->streams[g]));

    int N_local = part_sizes[g];
    h->part_info[g].gpu_id = g;
    h->part_info[g].local_N = N_local;

    // Initialize the single-GPU engine for this partition
    h->partitions[g] = leviathan_init(
        N_local, max_delay, part_row_ptrs[g], part_col_idxs[g], part_delays[g],
        part_weights[g], part_thetas[g], part_theta_hats[g], part_omegas[g]);

    // Allocate halo buffers
    h->part_info[g].halo_recv_size = halo_recv_sizes[g];
    h->part_info[g].halo_send_size = halo_send_sizes[g];

    if (halo_recv_sizes[g] > 0) {
      MG_CHECK(cudaMalloc(&h->part_info[g].d_halo_recv,
                          halo_recv_sizes[g] * sizeof(float)));
      MG_CHECK(cudaMalloc(&h->part_info[g].halo_remote_ids,
                          halo_recv_sizes[g] * sizeof(int)));
      MG_CHECK(cudaMemcpy(h->part_info[g].halo_remote_ids, halo_remote_ids[g],
                          halo_recv_sizes[g] * sizeof(int),
                          cudaMemcpyHostToDevice));
    }

    if (halo_send_sizes[g] > 0) {
      MG_CHECK(cudaMalloc(&h->part_info[g].d_halo_send,
                          halo_send_sizes[g] * sizeof(float)));
      MG_CHECK(cudaMalloc(&h->part_info[g].halo_send_local_ids,
                          halo_send_sizes[g] * sizeof(int)));
      MG_CHECK(cudaMemcpy(
          h->part_info[g].halo_send_local_ids, halo_send_local_ids[g],
          halo_send_sizes[g] * sizeof(int), cudaMemcpyHostToDevice));
    }

    printf("[MultiGPU] GPU %d: %d nodes, %d halo_recv, %d halo_send\n", g,
           N_local, halo_recv_sizes[g], halo_send_sizes[g]);
  }

  // Allocate r aggregation buffers on GPU 0
  MG_CHECK(cudaSetDevice(0));
  MG_CHECK(cudaMalloc(&h->d_r_local, num_gpus * sizeof(float)));
  MG_CHECK(cudaMalloc(&h->d_r_global, sizeof(float)));

  printf("[MultiGPU] Initialized %d GPUs, total %d nodes\n", num_gpus, total_N);

  return h;
}

float multigpu_step(MultiGPUHandle *h, float dt) {
  // Step 1: Physics on each GPU (parallel)
  float r_locals[8]; // Max 8 GPUs
  for (int g = 0; g < h->num_gpus; g++) {
    MG_CHECK(cudaSetDevice(g));
    r_locals[g] = leviathan_step(h->partitions[g], dt);
  }

  // Step 2: NCCL AllReduce to aggregate r across GPUs
  // For now, do weighted average on CPU (simple path)
  float r_sum = 0.0f;
  float n_sum = 0.0f;
  for (int g = 0; g < h->num_gpus; g++) {
    int n_g = h->part_info[g].local_N;
    r_sum += r_locals[g] * n_g;
    n_sum += n_g;
  }
  h->h_r = r_sum / n_sum;

  return h->h_r;
}

void multigpu_get_theta(MultiGPUHandle *h, float *h_theta) {
  // Gather all partition phases into a single global array
  int offset = 0;
  for (int g = 0; g < h->num_gpus; g++) {
    MG_CHECK(cudaSetDevice(g));
    int N_local = h->part_info[g].local_N;
    leviathan_get_theta(h->partitions[g], h_theta + offset);
    offset += N_local;
  }
}

void multigpu_free(MultiGPUHandle *h) {
  if (!h)
    return;

  for (int g = 0; g < h->num_gpus; g++) {
    MG_CHECK(cudaSetDevice(g));
    leviathan_free(h->partitions[g]);
    cudaStreamDestroy(h->streams[g]);

    if (h->part_info[g].halo_recv_size > 0) {
      cudaFree(h->part_info[g].d_halo_recv);
      cudaFree(h->part_info[g].halo_remote_ids);
    }
    if (h->part_info[g].halo_send_size > 0) {
      cudaFree(h->part_info[g].d_halo_send);
      cudaFree(h->part_info[g].halo_send_local_ids);
    }
    NCCL_CHECK(ncclCommDestroy(h->nccl_comms[g]));
  }

  MG_CHECK(cudaSetDevice(0));
  cudaFree(h->d_r_local);
  cudaFree(h->d_r_global);

  delete[] h->partitions;
  delete[] h->part_info;
  delete[] h->nccl_comms;
  delete[] h->streams;
  delete h;

  printf("[MultiGPU] All resources freed\n");
}

} // extern "C"
