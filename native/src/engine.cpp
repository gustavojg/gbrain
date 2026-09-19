// gBrain native engine — CPU backend
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <thread>

namespace gbrain {

namespace {
// Izhikevich presets, as in src/core/snn/neuron.ts
struct Preset { float a, b, c, d; };
constexpr Preset kRegularSpiking{0.02f, 0.2f, -65.0f, 8.0f};
constexpr Preset kFastSpiking{0.1f, 0.2f, -65.0f, 2.0f};
}  // namespace

// splitmix64: fast, decent, reproducible.
float Network::uniform() {
  rng_ += 0x9E3779B97F4A7C15ULL;
  uint64_t z = rng_;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<float>((z >> 40) * (1.0 / 16777216.0));
}

const char* Network::backend() {
#ifdef GBRAIN_CUDA
  return "cuda";
#else
  return "cpu";
#endif
}

Network::Network(const NetworkConfig& cfg) : cfg_(cfg), rng_(cfg.seed) {
  const uint32_t n = cfg_.neurons;
  threads_ = cfg_.threads > 0 ? cfg_.threads : std::max(1u, std::thread::hardware_concurrency());
  v_.assign(n, -65.0f);
  u_.resize(n);
  a_.resize(n); b_.resize(n); c_.resize(n); d_.resize(n);
  inhibitory_.assign(n, 0);
  input_.assign(n, 0.0f);
  inputNow_.assign(n, 0.0f);
  preTrace_.assign(n, 0.0f);
  postTrace_.assign(n, 0.0f);
  firedFlag_.assign(n, 0);
  // Interneurons: a fixed fraction, spread through the population.
  const uint32_t inhibitoryCount = static_cast<uint32_t>(n * cfg_.inhibitoryFraction);
  for (uint32_t i = 0; i < n; i++) {
    const bool inh = inhibitoryCount > 0 && (static_cast<uint64_t>(i) * inhibitoryCount / n) != (static_cast<uint64_t>(i + 1) * inhibitoryCount / n);
    const Preset& p = inh ? kFastSpiking : kRegularSpiking;
    inhibitory_[i] = inh ? 1 : 0;
    a_[i] = p.a; b_[i] = p.b; c_[i] = p.c; d_[i] = p.d;
    // Heterogeneous resting state: neurons that start identical and get the
    // same drive fire in lockstep forever; real membranes never do.
    v_[i] = -70.0f + 12.0f * uniform();
    u_[i] = p.b * v_[i];
  }
  threadInput_.assign(threads_, std::vector<float>(n, 0.0f));
  buildRandomSynapses();
}

Network::~Network() = default;

void Network::buildRandomSynapses() {
  const uint32_t n = cfg_.neurons;
  const uint32_t k = std::min(cfg_.fanIn, n > 1 ? n - 1 : 0u);
  // CSR by PRESYNAPTIC neuron: every neuron sends to k random targets (same
  // count out as in on average — a random graph of fan-in k).
  rowPtr_.resize(n + 1);
  targets_.resize(static_cast<size_t>(n) * k);
  weights_.resize(static_cast<size_t>(n) * k);
  for (uint32_t i = 0; i < n; i++) {
    rowPtr_[i] = i * k;
    for (uint32_t s = 0; s < k; s++) {
      uint32_t t = static_cast<uint32_t>(uniform() * n);
      if (t == i) t = (t + 1) % n;
      targets_[static_cast<size_t>(i) * k + s] = t;
      weights_[static_cast<size_t>(i) * k + s] = inhibitory_[i] ? cfg_.inhWeight : cfg_.excMin + uniform() * (cfg_.excMax - cfg_.excMin);
    }
  }
  rowPtr_[n] = n * k;
  buildTranspose();
}

void Network::setSynapses(std::vector<uint32_t> rowPtr, std::vector<uint32_t> targets, std::vector<float> weights) {
  rowPtr_ = std::move(rowPtr);
  targets_ = std::move(targets);
  weights_ = std::move(weights);
  buildTranspose();
}

// Transpose (CSC): for every postsynaptic neuron, the indices (into the CSR
// arrays) of the synapses that reach it.
void Network::buildTranspose() {
  const uint32_t n = cfg_.neurons;
  colPtr_.assign(n + 1, 0);
  const uint64_t m = rowPtr_[n];
  for (uint64_t s = 0; s < m; s++) colPtr_[targets_[s] + 1]++;
  for (uint32_t j = 0; j < n; j++) colPtr_[j + 1] += colPtr_[j];
  sources_.resize(m);
  synapseOfIncoming_.resize(m);
  std::vector<uint32_t> fill(colPtr_.begin(), colPtr_.end() - 1);
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t s = rowPtr_[i]; s < rowPtr_[i + 1]; s++) {
      const uint32_t j = targets_[s];
      const uint32_t slot = fill[j]++;
      sources_[slot] = i;
      synapseOfIncoming_[slot] = s;
    }
  }
}

void Network::resetState() {
  for (uint32_t i = 0; i < cfg_.neurons; i++) {
    v_[i] = -70.0f + 12.0f * uniform();
    u_[i] = b_[i] * v_[i];
  }
  std::fill(input_.begin(), input_.end(), 0.0f);
  std::fill(inputNow_.begin(), inputNow_.end(), 0.0f);
  std::fill(preTrace_.begin(), preTrace_.end(), 0.0f);
  std::fill(postTrace_.begin(), postTrace_.end(), 0.0f);
  fired_.clear();
  tick_ = 0;
}

void Network::parallelFor(uint32_t count, const std::function<void(uint32_t, uint32_t, uint32_t)>& body) {
  if (threads_ <= 1 || count < 4096) {
    body(0, 0, count);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(threads_);
  const uint32_t chunk = (count + threads_ - 1) / threads_;
  for (uint32_t t = 0; t < threads_; t++) {
    const uint32_t from = t * chunk, to = std::min(count, from + chunk);
    if (from >= to) break;
    pool.emplace_back([&, t, from, to] { body(t, from, to); });
  }
  for (auto& th : pool) th.join();
}

StepStats Network::step(const float* externalCurrent, float modulation) {
  const uint32_t n = cfg_.neurons;
  const float dt = cfg_.dt, halfDt = 0.5f * dt;
  const float decayPlus = std::exp(-dt / cfg_.tauPlus), decayMinus = std::exp(-dt / cfg_.tauMinus);
  tick_++;

  // What last tick's spikes delivered is what drives this tick.
  std::swap(input_, inputNow_);
  std::fill(input_.begin(), input_.end(), 0.0f);

  // 1. Neurons: integrate, detect spikes. Noise is drawn per neuron from a
  //    per-range hash so that threads never share the generator.
  const uint64_t tickSeed = rng_ ^ (static_cast<uint64_t>(tick_) * 0x9E3779B97F4A7C15ULL);
  const float noise = cfg_.noise;
  parallelFor(n, [&](uint32_t, uint32_t from, uint32_t to) {
    for (uint32_t i = from; i < to; i++) {
      uint64_t z = tickSeed + static_cast<uint64_t>(i) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z ^= z >> 31;
      const float jitter = noise * static_cast<float>((z >> 40) * (1.0 / 16777216.0));
      const float I = inputNow_[i] + (externalCurrent ? externalCurrent[i] : 0.0f) + jitter;
      float v = v_[i], u = u_[i];
      v += halfDt * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
      v += halfDt * (0.04f * v * v + 5.0f * v + 140.0f - u + I);
      u += dt * a_[i] * (b_[i] * v - u);
      uint8_t fired = 0;
      if (v >= 30.0f) {
        v = c_[i];
        u += d_[i];
        fired = 1;
      }
      v_[i] = v; u_[i] = u;
      firedFlag_[i] = fired;
      preTrace_[i] *= decayPlus;
      postTrace_[i] *= decayMinus;
    }
  });

  // 2. Collect the spikes (compact list, in index order).
  fired_.clear();
  StepStats stats;
  for (uint32_t i = 0; i < n; i++) {
    if (firedFlag_[i]) {
      fired_.push_back(i);
      if (inhibitory_[i]) stats.firedInhibitory++; else stats.firedExcitatory++;
    }
  }
  stats.fired = static_cast<uint32_t>(fired_.size());

  // 3. Deliver: each spike adds its synapses' weights to its targets' input
  //    for the next tick. Threads accumulate privately, then reduce.
  const uint32_t spikes = stats.fired;
  const bool plastic = cfg_.plastic && modulation != 0.0f;
  const float aMinus = cfg_.aMinus * modulation, aPlus = cfg_.aPlus * modulation, wMax = cfg_.wMax;
  if (spikes > 0) {
    const bool parallel = threads_ > 1 && spikes >= 256;
    parallelFor(parallel ? spikes : 0, [&](uint32_t t, uint32_t from, uint32_t to) {
      std::vector<float>& acc = threadInput_[t];
      for (uint32_t s = from; s < to; s++) {
        const uint32_t i = fired_[s];
        for (uint32_t k = rowPtr_[i]; k < rowPtr_[i + 1]; k++) acc[targets_[k]] += weights_[k];
      }
    });
    if (parallel) {
      parallelFor(n, [&](uint32_t, uint32_t from, uint32_t to) {
        for (uint32_t t = 0; t < threads_; t++) {
          std::vector<float>& acc = threadInput_[t];
          for (uint32_t j = from; j < to; j++) { input_[j] += acc[j]; acc[j] = 0.0f; }
        }
      });
    } else {
      for (uint32_t s = 0; s < spikes; s++) {
        const uint32_t i = fired_[s];
        for (uint32_t k = rowPtr_[i]; k < rowPtr_[i + 1]; k++) input_[targets_[k]] += weights_[k];
      }
    }

    // 4. Plasticity (excitatory synapses only). Pre spike: depress its
    //    synapses by the targets' post traces; post spike: potentiate the
    //    synapses into it by their sources' pre traces. Then bump the traces.
    if (plastic) {
      for (uint32_t s = 0; s < spikes; s++) {
        const uint32_t i = fired_[s];
        if (!inhibitory_[i]) {
          for (uint32_t k = rowPtr_[i]; k < rowPtr_[i + 1]; k++) {
            float w = weights_[k] - aMinus * postTrace_[targets_[k]];
            weights_[k] = w < 0.0f ? 0.0f : w;
          }
        }
        for (uint32_t k = colPtr_[i]; k < colPtr_[i + 1]; k++) {
          const uint32_t src = sources_[k];
          if (inhibitory_[src]) continue;
          const uint32_t syn = synapseOfIncoming_[k];
          float w = weights_[syn] + aPlus * preTrace_[src];
          weights_[syn] = w > wMax ? wMax : w;
        }
      }
      for (uint32_t s = 0; s < spikes; s++) {
        const uint32_t i = fired_[s];
        preTrace_[i] += 1.0f;
        postTrace_[i] += 1.0f;
      }
    }
  }
  return stats;
}

}  // namespace gbrain
