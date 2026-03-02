// leviathan_pybind11.cpp
// Pybind11 bindings for CUDA phase dynamics engine

#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

class LeviathanPython {
public:
  LeviathanHandle *handle;
  int N;

  LeviathanPython(int N_, int max_delay, py::array_t<int> row_ptr,
                  py::array_t<int> col_idx, py::array_t<uint8_t> delays,
                  py::array_t<float> weights, py::array_t<float> theta,
                  py::array_t<float> theta_hat, py::array_t<float> omega)
      : N(N_) {
    handle = leviathan_init(
        N_, max_delay, (int *)row_ptr.request().ptr,
        (int *)col_idx.request().ptr, (uint8_t *)delays.request().ptr,
        (float *)weights.request().ptr, (float *)theta.request().ptr,
        (float *)theta_hat.request().ptr, (float *)omega.request().ptr);
  }

  float step(float dt) {
    py::gil_scoped_release release; // Release GIL during GPU work
    return leviathan_step(handle, dt);
  }

  // Device-to-host copy into a new NumPy array
  py::array_t<float> get_theta() {
    auto result = py::array_t<float>(N);
    auto buf = result.request();
    leviathan_get_theta(handle, (float *)buf.ptr);
    return result;
  }

  void set_theta(py::array_t<float> theta) {
    auto buf = theta.request();
    if (buf.size != N) {
      throw std::runtime_error("theta array size mismatch: expected " +
                               std::to_string(N) + ", got " +
                               std::to_string(buf.size));
    }
    leviathan_set_theta(handle, (float *)buf.ptr);
  }

  ~LeviathanPython() { leviathan_free(handle); }
};

PYBIND11_MODULE(leviathan_cuda, m) {
  m.doc() = "GPU-accelerated delayed Kuramoto network with adaptive coupling";

  py::class_<LeviathanPython>(m, "LeviathanEngine")
      .def(
          py::init<int, int, py::array_t<int>, py::array_t<int>,
                   py::array_t<uint8_t>, py::array_t<float>, py::array_t<float>,
                   py::array_t<float>, py::array_t<float>>())
      .def("step", &LeviathanPython::step,
           "Execute one integration step, returns order parameter r")
      .def("get_theta", &LeviathanPython::get_theta,
           "Copy current phase array from device to host (returns NumPy array)")
      .def("set_theta", &LeviathanPython::set_theta,
           "Upload phase array from host to device");
}