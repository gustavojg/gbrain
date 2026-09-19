// gBrain native engine — CUDA backend
// ============================================================================
// The same step as engine.cpp, as GPU kernels. Built only with -DGBRAIN_CUDA
// (CMake option) on a machine with the CUDA toolkit; this file has been
// written to the CPU backend's semantics but NOT yet compiled or run — the
// development machine has no NVIDIA GPU. First run it on a rented GPU with
// `gbrain-bench` and compare its numbers and firing rates with the CPU's.
//
// Layout: the same structure-of-arrays state as the CPU backend, mirrored on
// the device. One thread per neuron for integration; one thread per spike
// for delivery (atomicAdd into the synaptic currents, excitatory or
// inhibitory by the source), one per spike for plasticity (atomic weight
// updates). Synaptic currents decay exponentially as in the CPU backend.
#ifdef GBRAIN_CUDA
#include <cuda_runtime.h>

#include <cstdint>

namespace gbrain {
namespace cuda {

__global__ void integrate(uint32_t n, float dt, float noise, uint64_t tickSeed,
                          float* synExc, float* synInh, float decayExc, float decayInh, float* resource, float recover,
                          const float* external,
                          float* v, float* u, const float* a, const float* b, const float* c, const float* d,
                          uint8_t* fired, float* preTrace, float* postTrace, float decayPlus, float decayMinus) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t z = tickSeed + static_cast<uint64_t>(i) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  const float jitter = noise * static_cast<float>((z >> 40) * (1.0 / 16777216.0));
  const float I = synExc[i] + synInh[i] + (external ? external[i] : 0.0f) + jitter;
  synExc[i] *= decayExc;
  synInh[i] *= decayInh;
  resource[i] += (1.0f - resource[i]) * recover;
  const float halfDt = 0.5f * dt;
  float vv = v[i], uu = u[i];
  vv += halfDt * (0.04f * vv * vv + 5.0f * vv + 140.0f - uu + I);
  vv += halfDt * (0.04f * vv * vv + 5.0f * vv + 140.0f - uu + I);
  uu += dt * a[i] * (b[i] * vv - uu);
  uint8_t f = 0;
  if (vv >= 30.0f) { vv = c[i]; uu += d[i]; f = 1; }
  v[i] = vv; u[i] = uu; fired[i] = f;
  preTrace[i] *= decayPlus;
  postTrace[i] *= decayMinus;
}

/// Compacts the fired flags into a spike list (order not guaranteed).
__global__ void collect(uint32_t n, const uint8_t* fired, uint32_t* list, uint32_t* count) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || !fired[i]) return;
  const uint32_t slot = atomicAdd(count, 1u);
  list[slot] = i;
}

/// One thread per spike: its synapses' weights land in the targets' synaptic current (by source type).
__global__ void deliver(uint32_t spikes, const uint32_t* list, const uint8_t* inhibitory, const uint32_t* rowPtr,
                        const uint32_t* targets, const float* weights, float* synExc, float* synInh,
                        float* resource, float keep) {
  const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= spikes) return;
  const uint32_t i = list[s];
  float* dst = inhibitory[i] ? synInh : synExc;
  // Short-term depression: an excitatory spike delivers what its resources allow, and spends a share of them.
  const float efficacy = inhibitory[i] ? 1.0f : resource[i];
  for (uint32_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) atomicAdd(&dst[targets[k]], weights[k] * efficacy);
  if (!inhibitory[i]) resource[i] *= keep;
}

/// Depression on the spiking neuron's outgoing excitatory synapses; potentiation on its incoming ones.
__global__ void plasticity(uint32_t spikes, const uint32_t* list, const uint8_t* inhibitory,
                           const uint32_t* rowPtr, const uint32_t* targets, float* weights,
                           const uint32_t* colPtr, const uint32_t* sources, const uint32_t* synapseOfIncoming,
                           const float* preTrace, const float* postTrace, float aPlus, float aMinus, float wMax) {
  const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= spikes) return;
  const uint32_t i = list[s];
  if (!inhibitory[i]) {
    for (uint32_t k = rowPtr[i]; k < rowPtr[i + 1]; k++) {
      const float w = weights[k] - aMinus * postTrace[targets[k]];
      weights[k] = w < 0.0f ? 0.0f : w;
    }
  }
  for (uint32_t k = colPtr[i]; k < colPtr[i + 1]; k++) {
    const uint32_t src = sources[k];
    if (inhibitory[src]) continue;
    const uint32_t syn = synapseOfIncoming[k];
    const float w = weights[syn] + aPlus * preTrace[src];
    weights[syn] = w > wMax ? wMax : w;
  }
}

/// The traces of the neurons that fired grow by one.
__global__ void bumpTraces(uint32_t spikes, const uint32_t* list, float* preTrace, float* postTrace) {
  const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= spikes) return;
  preTrace[list[s]] += 1.0f;
  postTrace[list[s]] += 1.0f;
}

}  // namespace cuda
}  // namespace gbrain
#endif
