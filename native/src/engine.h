// gBrain native engine — spiking network core
// ============================================================================
// The hot loop of the brain, written to scale to a million neurons:
//
//   - Izhikevich neurons in structure-of-arrays form (v, u, a, b, c, d),
//     integrated exactly as the TypeScript model does (two Euler half-steps
//     of dt for v, one step for u, threshold 30 mV, reset to c, u += d).
//   - Sparse synapses in compressed-sparse-row form by presynaptic neuron
//     (targets + weights), with the transpose kept as an index map so that
//     the synapses INTO a neuron can be walked too (for potentiation).
//   - Real inhibitory interneurons (fast-spiking, negative outgoing weights)
//     instead of an algorithmic k-winners-take-all: competition is what the
//     inhibitory population does to the excitatory one.
//   - Spike-timing-dependent plasticity by pre/post traces (Morrison, Diesmann
//     & Gerstner 2008), on excitatory synapses only, gated by a global
//     modulation factor (the neuromodulators' say), weights clipped to
//     [0, wmax].
//   - Spikes propagate with a one-tick delay: what fires at tick t drives its
//     targets at tick t+1.
//   - Multithreaded on CPU over neuron ranges; the same kernels exist for
//     CUDA (engine_cuda.cu), selected at build time.
//
// The engine knows nothing of regions, buses or memories: it is the substrate
// those are built on. The brain (TypeScript) drives it through the Node addon.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gbrain {

struct NetworkConfig {
  uint32_t neurons = 10000;
  /// Synapses per neuron (fan-in), drawn at random when the network is built.
  uint32_t fanIn = 100;
  /// Fraction of neurons that are inhibitory interneurons (fast spiking).
  float inhibitoryFraction = 0.2f;
  /// Initial excitatory weight range and the inhibitory weight (negative).
  float excMin = 0.0f, excMax = 0.5f;
  float inhWeight = -1.0f;
  /// Integration step (ms), as the brain's dt.
  float dt = 1.0f;
  /// Background current noise amplitude (uniform 0..noise), as the TS network adds.
  float noise = 0.05f;
  /// STDP
  bool plastic = true;
  float aPlus = 0.01f, aMinus = 0.012f;
  float tauPlus = 20.0f, tauMinus = 20.0f;
  float wMax = 1.0f;
  /// Threads for the CPU backend (0 = hardware concurrency).
  uint32_t threads = 0;
  uint64_t seed = 0x5eed;
};

struct StepStats {
  uint32_t fired = 0;
  uint32_t firedExcitatory = 0;
  uint32_t firedInhibitory = 0;
};

class Network {
 public:
  explicit Network(const NetworkConfig& cfg);
  ~Network();

  /// One tick: external currents (length = neurons, may be null = zeros) plus
  /// the synaptic input from last tick's spikes drive every neuron.
  StepStats step(const float* externalCurrent, float modulation = 1.0f);

  /// Neurons that fired on the last step (indices).
  const std::vector<uint32_t>& fired() const { return fired_; }

  uint32_t neurons() const { return cfg_.neurons; }
  uint64_t synapses() const { return static_cast<uint64_t>(rowPtr_.back()); }
  bool isInhibitory(uint32_t n) const { return inhibitory_[n] != 0; }
  const NetworkConfig& config() const { return cfg_; }

  /// Membrane potentials and weights, for inspection and persistence.
  const std::vector<float>& potentials() const { return v_; }
  const std::vector<float>& weights() const { return weights_; }
  std::vector<float>& weights() { return weights_; }
  const std::vector<uint32_t>& rowPtr() const { return rowPtr_; }
  const std::vector<uint32_t>& targets() const { return targets_; }

  /// Replaces the connectivity (CSR by presynaptic neuron). Weights of
  /// inhibitory rows are taken as given (should be ≤ 0).
  void setSynapses(std::vector<uint32_t> rowPtr, std::vector<uint32_t> targets, std::vector<float> weights);

  /// Resets membrane state and traces (not the weights).
  void resetState();

  /// Name of the backend in use ("cpu" or "cuda").
  static const char* backend();

 private:
  void buildRandomSynapses();
  void buildTranspose();
  void parallelFor(uint32_t count, const std::function<void(uint32_t, uint32_t, uint32_t)>& body);

  NetworkConfig cfg_;
  // Neuron state (structure of arrays).
  std::vector<float> v_, u_, a_, b_, c_, d_;
  std::vector<uint8_t> inhibitory_;
  std::vector<float> input_;      // synaptic input accumulated for the NEXT tick
  std::vector<float> inputNow_;   // synaptic input being consumed this tick
  std::vector<float> preTrace_, postTrace_;
  std::vector<uint8_t> firedFlag_;
  std::vector<uint32_t> fired_;
  // Synapses: CSR by presynaptic neuron, and its transpose as index map.
  std::vector<uint32_t> rowPtr_, targets_;
  std::vector<float> weights_;
  std::vector<uint32_t> colPtr_, sources_, synapseOfIncoming_;
  // Per-thread accumulation buffers for spike delivery.
  std::vector<std::vector<float>> threadInput_;
  uint32_t threads_ = 1;
  uint64_t rng_;
  float uniform();
  uint32_t tick_ = 0;
};

}  // namespace gbrain
