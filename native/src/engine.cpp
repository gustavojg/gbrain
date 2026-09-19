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
  synExc_.assign(n, 0.0f);
  synInh_.assign(n, 0.0f);
  resource_.assign(n, 1.0f);
  preTrace_.assign(n, 0.0f);
  postTrace_.assign(n, 0.0f);
  firedFlag_.assign(n, 0);
  spikeCount_.assign(n, 0);
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
  threadInput_.assign(threads_, std::vector<float>(static_cast<size_t>(n) * 2, 0.0f));
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
      const float exc = (cfg_.excMin + uniform() * (cfg_.excMax - cfg_.excMin)) * (inhibitory_[t] ? cfg_.excToInhGain : 1.0f);
      weights_[static_cast<size_t>(i) * k + s] = inhibitory_[i] ? cfg_.inhWeight : exc;
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

void Network::setInputProjection(uint32_t channels, std::vector<uint32_t> rowPtr, std::vector<uint32_t> cols, std::vector<float> weights) {
  inputChannels_ = channels;
  inRowPtr_ = std::move(rowPtr);
  inCols_ = std::move(cols);
  inWeights_ = std::move(weights);
  channelCurrent_.assign(cfg_.neurons, 0.0f);
}

void Network::buildRandomInputProjection(uint32_t channels, uint32_t fanIn, float wMin, float wMax) {
  const uint32_t n = cfg_.neurons;
  const uint32_t k = std::min(fanIn, channels);
  std::vector<uint32_t> rowPtr(n + 1);
  std::vector<uint32_t> cols(static_cast<size_t>(n) * k);
  std::vector<float> weights(static_cast<size_t>(n) * k);
  for (uint32_t i = 0; i < n; i++) {
    rowPtr[i] = i * k;
    for (uint32_t s = 0; s < k; s++) {
      cols[static_cast<size_t>(i) * k + s] = static_cast<uint32_t>(uniform() * channels);
      weights[static_cast<size_t>(i) * k + s] = wMin + uniform() * (wMax - wMin);
    }
  }
  rowPtr[n] = n * k;
  setInputProjection(channels, std::move(rowPtr), std::move(cols), std::move(weights));
}

StepStats Network::stepChannels(const float* channels, float modulation) {
  if (inputChannels_ == 0) return step(nullptr, modulation);
  const uint32_t n = cfg_.neurons;
  parallelFor(n, [&](uint32_t, uint32_t from, uint32_t to) {
    for (uint32_t i = from; i < to; i++) {
      float I = 0.0f;
      for (uint32_t k = inRowPtr_[i]; k < inRowPtr_[i + 1]; k++) I += inWeights_[k] * channels[inCols_[k]];
      channelCurrent_[i] = I;
    }
  });
  return step(channelCurrent_.data(), modulation);
}

void Network::resetState() {
  for (uint32_t i = 0; i < cfg_.neurons; i++) {
    v_[i] = -70.0f + 12.0f * uniform();
    u_[i] = b_[i] * v_[i];
  }
  std::fill(synExc_.begin(), synExc_.end(), 0.0f);
  std::fill(synInh_.begin(), synInh_.end(), 0.0f);
  std::fill(resource_.begin(), resource_.end(), 1.0f);
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
  // Synaptic currents fade with their time constants (0 = a one-tick pulse).
  const float decayExc = cfg_.tauSynExc > 0.0f ? std::exp(-dt / cfg_.tauSynExc) : 0.0f;
  const float decayInh = cfg_.tauSynInh > 0.0f ? std::exp(-dt / cfg_.tauSynInh) : 0.0f;
  // Short-term depression: resources recover toward 1 each tick, and a spike spends a share of them.
  const bool depressing = cfg_.shortTermDepression && cfg_.stdTauRec > 0.0f;
  const float recover = depressing ? dt / cfg_.stdTauRec : 0.0f, keep = depressing ? 1.0f - cfg_.stdU : 1.0f;
  tick_++;

  // 1. Neurons: the synaptic currents (what earlier spikes delivered, fading)
  //    plus the external current drive the integration; then the currents
  //    decay, and this tick's spikes are added to them below. Noise is drawn
  //    per neuron from a per-range hash so that threads never share the
  //    generator.
  const uint64_t tickSeed = rng_ ^ (static_cast<uint64_t>(tick_) * 0x9E3779B97F4A7C15ULL);
  const float noise = cfg_.noise;
  parallelFor(n, [&](uint32_t, uint32_t from, uint32_t to) {
    for (uint32_t i = from; i < to; i++) {
      uint64_t z = tickSeed + static_cast<uint64_t>(i) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z ^= z >> 31;
      const float jitter = noise * static_cast<float>((z >> 40) * (1.0 / 16777216.0));
      const float I = synExc_[i] + synInh_[i] + (externalCurrent ? externalCurrent[i] : 0.0f) + jitter;
      synExc_[i] *= decayExc;
      synInh_[i] *= decayInh;
      if (depressing) resource_[i] += (1.0f - resource_[i]) * recover;
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
  if (cfg_.structural) for (uint32_t s = 0; s < stats.fired; s++) spikeCount_[fired_[s]]++;

  // 3. Deliver: each spike adds its synapses' weights to its targets'
  //    synaptic current (excitatory or inhibitory by the source), felt from
  //    the next tick. Threads accumulate privately, then reduce.
  const uint32_t spikes = stats.fired;
  const bool plastic = cfg_.plastic && modulation != 0.0f;
  const float aMinus = cfg_.aMinus * modulation, aPlus = cfg_.aPlus * modulation, wMax = cfg_.wMax;
  if (spikes > 0) {
    const bool parallel = threads_ > 1 && spikes >= 256;
    parallelFor(parallel ? spikes : 0, [&](uint32_t t, uint32_t from, uint32_t to) {
      std::vector<float>& acc = threadInput_[t];
      for (uint32_t s = from; s < to; s++) {
        const uint32_t i = fired_[s];
        const uint32_t off = inhibitory_[i] ? n : 0;
        const float efficacy = inhibitory_[i] ? 1.0f : resource_[i];
        for (uint32_t k = rowPtr_[i]; k < rowPtr_[i + 1]; k++) acc[off + targets_[k]] += weights_[k] * efficacy;
        if (!inhibitory_[i]) resource_[i] *= keep;
      }
    });
    if (parallel) {
      parallelFor(n, [&](uint32_t, uint32_t from, uint32_t to) {
        for (uint32_t t = 0; t < threads_; t++) {
          std::vector<float>& acc = threadInput_[t];
          for (uint32_t j = from; j < to; j++) {
            synExc_[j] += acc[j]; acc[j] = 0.0f;
            synInh_[j] += acc[n + j]; acc[n + j] = 0.0f;
          }
        }
      });
    } else {
      for (uint32_t s = 0; s < spikes; s++) {
        const uint32_t i = fired_[s];
        float* dst = inhibitory_[i] ? synInh_.data() : synExc_.data();
        const float efficacy = inhibitory_[i] ? 1.0f : resource_[i];
        for (uint32_t k = rowPtr_[i]; k < rowPtr_[i + 1]; k++) dst[targets_[k]] += weights_[k] * efficacy;
        if (!inhibitory_[i]) resource_[i] *= keep;
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
      // Inhibitory plasticity on I→E synapses (Vogels 2011). Pre (interneuron)
      // spike: the synapse grows by how far the target's post trace exceeds
      // the target (alpha = 2 · rate · tau); post (excitatory) spike: it grows
      // by the interneuron's pre trace. Inhibitory weights are negative, so
      // "grows" means more negative, bounded at −inhMax.
      if (cfg_.inhibitoryPlasticity) {
        const float eta = cfg_.iEta, alpha = 2.0f * cfg_.targetRate * cfg_.tauMinus, inhMax = cfg_.inhMax;
        for (uint32_t s = 0; s < spikes; s++) {
          const uint32_t i = fired_[s];
          if (inhibitory_[i]) {
            for (uint32_t k = rowPtr_[i]; k < rowPtr_[i + 1]; k++) {
              const uint32_t t = targets_[k];
              if (inhibitory_[t]) continue;
              float w = weights_[k] - eta * (postTrace_[t] - alpha);
              weights_[k] = w < -inhMax ? -inhMax : (w > 0.0f ? 0.0f : w);
            }
          } else {
            for (uint32_t k = colPtr_[i]; k < colPtr_[i + 1]; k++) {
              const uint32_t src = sources_[k];
              if (!inhibitory_[src]) continue;
              const uint32_t syn = synapseOfIncoming_[k];
              float w = weights_[syn] - eta * preTrace_[src];
              weights_[syn] = w < -inhMax ? -inhMax : w;
            }
          }
        }
      }
      for (uint32_t s = 0; s < spikes; s++) {
        const uint32_t i = fired_[s];
        preTrace_[i] += 1.0f;
        postTrace_[i] += 1.0f;
      }
    }
  }
  // 5. Structural plasticity, now and then.
  if (cfg_.structural && cfg_.rewireEvery > 0 && tick_ % cfg_.rewireEvery == 0) rewire();
  return stats;
}

// Synaptogenesis between coactive neurons. Neurons that fired at least
// `coactiveSpikes` times since the last rewiring form the coactive set; each
// excitatory member gives up its weakest synapses (below `pruneBelow`) and
// grows new ones onto other members it is not yet connected to. Fan-in stays
// fixed (slots are reused), so the memory does not grow. The transpose is
// rebuilt afterwards; the traces of the new synapses start at zero.
void Network::rewire() {
  const uint32_t n = cfg_.neurons;
  std::vector<uint32_t> coactive;
  for (uint32_t i = 0; i < n; i++) {
    if (spikeCount_[i] >= cfg_.coactiveSpikes && !inhibitory_[i]) coactive.push_back(i);
  }
  std::fill(spikeCount_.begin(), spikeCount_.end(), 0u);
  if (coactive.size() < 2) return;
  bool changed = false;
  for (uint32_t pre : coactive) {
    for (uint32_t r = 0; r < cfg_.rewiresPerEvent; r++) {
      // The weakest synapse of the row, if it is weak enough to give up.
      uint32_t weakest = rowPtr_[pre];
      for (uint32_t k = rowPtr_[pre]; k < rowPtr_[pre + 1]; k++) if (weights_[k] < weights_[weakest]) weakest = k;
      if (rowPtr_[pre + 1] == rowPtr_[pre] || weights_[weakest] >= cfg_.pruneBelow) break;
      // A coactive partner it does not reach yet.
      uint32_t target = n;
      for (uint32_t attempt = 0; attempt < 8 && target == n; attempt++) {
        const uint32_t candidate = coactive[static_cast<uint32_t>(uniform() * coactive.size())];
        if (candidate == pre) continue;
        bool already = false;
        for (uint32_t k = rowPtr_[pre]; k < rowPtr_[pre + 1]; k++) if (targets_[k] == candidate) { already = true; break; }
        if (!already) target = candidate;
      }
      if (target == n) break;
      targets_[weakest] = target;
      weights_[weakest] = cfg_.newWeight;
      rewired_++;
      changed = true;
    }
  }
  if (changed) buildTranspose();
}

}  // namespace gbrain
