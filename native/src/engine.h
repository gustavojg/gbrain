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
//   - Spikes propagate with a one-tick delay and land as exponential synaptic
//     currents: what fires at tick t drives its targets from tick t+1, fading
//     with an AMPA-like time constant for excitatory synapses and a
//     GABA_A-like one for inhibitory ones. That is what lets inputs sum in
//     time (a recurrent assembly cannot hold itself up on one-tick pulses).
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
  /// Gain on excitatory synapses onto interneurons: E→I coupling is dense and strong in cortex,
  /// and it is what makes feedback inhibition track the excitatory activity.
  float excToInhGain = 1.0f;
  /// Integration step (ms), as the brain's dt.
  float dt = 1.0f;
  /// Synaptic current time constants (ms): excitatory (AMPA ≈ 5 ms) and
  /// inhibitory (GABA_A ≈ 10 ms). 0 = a one-tick pulse, no temporal summation.
  float tauSynExc = 5.0f, tauSynInh = 10.0f;
  /// Short-term synaptic depression on excitatory synapses (Tsodyks & Markram
  /// 1997), by presynaptic neuron: each spike spends a fraction `stdU` of the
  /// neuron's synaptic resources, which recover with `stdTauRec` (ms). It is
  /// what makes a cell assembly transient — it ignites and fades — instead of
  /// a runaway attractor that outlives its input and answers to everything.
  bool shortTermDepression = true;
  float stdU = 0.3f;
  float stdTauRec = 200.0f;
  /// Background current noise amplitude (uniform 0..noise), as the TS network adds.
  float noise = 0.05f;
  /// STDP
  bool plastic = true;
  float aPlus = 0.01f, aMinus = 0.012f;
  float tauPlus = 20.0f, tauMinus = 20.0f;
  float wMax = 1.0f;
  /// Inhibitory plasticity (Vogels et al. 2011): the synapses from
  /// interneurons onto an excitatory neuron grow when it fires above a target
  /// rate and shrink when below, so that inhibition comes to balance each
  /// neuron's excitation — what lets recurrent excitation be strong without
  /// runaway. `targetRate` is in spikes per tick; the traces are the STDP ones.
  bool inhibitoryPlasticity = false;
  float iEta = 0.002f;
  float targetRate = 0.05f;
  float inhMax = 8.0f;
  /// Structural plasticity: every `rewireEvery` ticks, each excitatory neuron
  /// that has been active in the window swaps up to `rewiresPerEvent` of its
  /// weakest synapses (below `pruneBelow`) for new ones onto neurons that were
  /// active in the same window (synaptogenesis between coactive neurons;
  /// Holtmaat & Svoboda 2009), born at `newWeight`.
  bool structural = false;
  uint32_t rewireEvery = 50;
  uint32_t coactiveSpikes = 2;
  uint32_t rewiresPerEvent = 2;
  float pruneBelow = 0.05f;
  float newWeight = 0.3f;
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

  /// The afferent projection: input channels → neurons, sparse (CSR by
  /// neuron: which channels each neuron listens to, with what weight). What
  /// a region receives from the rest of the brain enters through it.
  void setInputProjection(uint32_t channels, std::vector<uint32_t> rowPtr, std::vector<uint32_t> cols, std::vector<float> weights);
  /// A random afferent projection: every neuron listens to `fanIn` channels with weights in [wMin, wMax].
  void buildRandomInputProjection(uint32_t channels, uint32_t fanIn, float wMin, float wMax);
  uint32_t inputChannels() const { return inputChannels_; }

  /// One tick driven by input CHANNELS (length = inputChannels): the external
  /// current of each neuron is its projection's weighted sum of the channels.
  StepStats stepChannels(const float* channels, float modulation = 1.0f);

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

  /// Synapses rewired so far (structural plasticity).
  uint64_t rewired() const { return rewired_; }

  /// Name of the backend in use ("cpu" or "cuda").
  static const char* backend();

 private:
  void buildRandomSynapses();
  void buildTranspose();
  void rewire();
  void parallelFor(uint32_t count, const std::function<void(uint32_t, uint32_t, uint32_t)>& body);

  NetworkConfig cfg_;
  // Neuron state (structure of arrays).
  std::vector<float> v_, u_, a_, b_, c_, d_;
  std::vector<uint8_t> inhibitory_;
  std::vector<float> synExc_, synInh_;  // synaptic currents (exponential), by source type
  std::vector<float> resource_;         // short-term depression: synaptic resources per presynaptic neuron (1 = rested)
  std::vector<float> preTrace_, postTrace_;
  std::vector<uint8_t> firedFlag_;
  std::vector<uint32_t> fired_;
  // Synapses: CSR by presynaptic neuron, and its transpose as index map.
  std::vector<uint32_t> rowPtr_, targets_;
  std::vector<float> weights_;
  std::vector<uint32_t> colPtr_, sources_, synapseOfIncoming_;
  // Afferent projection (CSR by neuron over input channels).
  uint32_t inputChannels_ = 0;
  std::vector<uint32_t> inRowPtr_, inCols_;
  std::vector<float> inWeights_;
  std::vector<float> channelCurrent_;
  // Per-thread accumulation buffers for spike delivery ([0, n) excitatory sources, [n, 2n) inhibitory).
  std::vector<std::vector<float>> threadInput_;
  uint32_t threads_ = 1;
  uint64_t rng_;
  float uniform();
  uint32_t tick_ = 0;
  // Structural plasticity: spikes per neuron since the last rewiring, and the count of rewirings.
  std::vector<uint32_t> spikeCount_;
  uint64_t rewired_ = 0;
};

}  // namespace gbrain
