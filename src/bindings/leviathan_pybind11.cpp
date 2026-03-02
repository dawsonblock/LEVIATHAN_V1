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
void leviathan_set_gain(LeviathanHandle *handle, float g);
float leviathan_get_gain(LeviathanHandle *handle);
void leviathan_set_gain_controller(LeviathanHandle *handle, bool enabled);
size_t leviathan_get_vram_usage(LeviathanHandle *handle);
void leviathan_reset_weights(LeviathanHandle *handle, float *h_weights);
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

  void set_gain(float g) { leviathan_set_gain(handle, g); }
  float get_gain() { return leviathan_get_gain(handle); }
  void set_gain_controller(bool enabled) {
    leviathan_set_gain_controller(handle, enabled);
  }
  size_t get_vram_usage() { return leviathan_get_vram_usage(handle); }

  void reset_weights(py::array_t<float> weights) {
    auto buf = weights.request();
    leviathan_reset_weights(handle, (float *)buf.ptr);
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
           "Upload phase array from host to device")
      .def("set_gain", &LeviathanPython::set_gain, "Set coupling gain g")
      .def("get_gain", &LeviathanPython::get_gain,
           "Get current coupling gain g")
      .def("set_gain_controller", &LeviathanPython::set_gain_controller,
           "Toggle homeostatic gain controller")
      .def("get_vram_usage", &LeviathanPython::get_vram_usage,
           "Get total VRAM consumed by this engine in bytes")
      .def("reset_weights", &LeviathanPython::reset_weights,
           "Reload synaptic weights from host array");
}