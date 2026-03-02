// leviathan_phi_pybind.cpp
// Pybind11 bindings for the GPU Φ solver

#include <cstdint>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" {
struct PhiHandle;
PhiHandle *phi_init(int num_hubs, int *h_hub_indices);
void phi_accumulate(PhiHandle *handle, const float *d_theta);
float phi_compute(PhiHandle *handle);
void phi_reset(PhiHandle *handle);
int64_t phi_get_transition_count(PhiHandle *handle);
void phi_free(PhiHandle *handle);
}

class GPUPhiWorker {
public:
  PhiHandle *handle;
  int num_hubs;
  int theta_capacity;
  float *d_theta_tmp; // Persistent device buffer

  GPUPhiWorker(py::array_t<int> hub_indices)
      : d_theta_tmp(nullptr), theta_capacity(0) {
    auto buf = hub_indices.request();
    num_hubs = static_cast<int>(buf.size);
    handle = phi_init(num_hubs, (int *)buf.ptr);
    if (!handle) {
      throw std::runtime_error("Failed to initialize GPU Phi worker");
    }
  }

  void accumulate_from_host(py::array_t<float> theta) {
    py::gil_scoped_release release;

    auto buf = theta.request();
    float *h_theta = (float *)buf.ptr;
    int N = static_cast<int>(buf.size);

    // Lazy-allocate persistent buffer (once, or if N changes)
    if (N > theta_capacity) {
      if (d_theta_tmp)
        cudaFree(d_theta_tmp);
      cudaMalloc(&d_theta_tmp, N * sizeof(float));
      theta_capacity = N;
    }

    cudaMemcpy(d_theta_tmp, h_theta, N * sizeof(float), cudaMemcpyHostToDevice);
    phi_accumulate(handle, d_theta_tmp);
  }

  float compute() { return phi_compute(handle); }

  void reset() { phi_reset(handle); }

  int64_t get_transition_count() { return phi_get_transition_count(handle); }

  ~GPUPhiWorker() {
    if (d_theta_tmp)
      cudaFree(d_theta_tmp);
    if (handle)
      phi_free(handle);
  }
};

PYBIND11_MODULE(leviathan_phi, m) {
  m.doc() = "GPU-accelerated Φ (integrated information) solver";

  py::class_<GPUPhiWorker>(m, "GPUPhiWorker")
      .def(py::init<py::array_t<int>>(), py::arg("hub_indices"),
           "Initialize Φ worker with hub node indices")
      .def("accumulate", &GPUPhiWorker::accumulate_from_host, py::arg("theta"),
           "Accumulate TPM transition from phase snapshot")
      .def("compute", &GPUPhiWorker::compute, "Compute Φ from accumulated TPM")
      .def("reset", &GPUPhiWorker::reset, "Reset TPM for new window")
      .def("get_transition_count", &GPUPhiWorker::get_transition_count,
           "Get accumulated transition count");
}
