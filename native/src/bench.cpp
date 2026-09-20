// gBrain native engine — benchmark
// ============================================================================
// Runs a random network for a number of ticks and reports ticks per second,
// spikes per second and the mean firing rate. This is the number that
// decides where the million-neuron brain runs.
//
//   gbrain-bench [neurons] [fanIn] [ticks] [threads] [drive]
//
// `drive` is a constant external current given to a random 10% of the
// excitatory neurons (a stimulus), so that the network is not silent.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine.h"

int main(int argc, char** argv) {
  gbrain::NetworkConfig cfg;
  cfg.neurons = argc > 1 ? static_cast<uint32_t>(std::atol(argv[1])) : 10000;
  cfg.fanIn = argc > 2 ? static_cast<uint32_t>(std::atol(argv[2])) : 100;
  const uint32_t ticks = argc > 3 ? static_cast<uint32_t>(std::atol(argv[3])) : 1000;
  cfg.threads = argc > 4 ? static_cast<uint32_t>(std::atol(argv[4])) : 0;
  const float drive = argc > 5 ? static_cast<float>(std::atof(argv[5])) : 8.0f;

  auto t0 = std::chrono::steady_clock::now();
  gbrain::Network net(cfg);
  auto t1 = std::chrono::steady_clock::now();
  const double buildSeconds = std::chrono::duration<double>(t1 - t0).count();
  std::printf("gbrain-bench: %u neurons, fan-in %u, %llu synapses, %s backend, %u threads — built in %.2f s\n",
              net.neurons(), cfg.fanIn, static_cast<unsigned long long>(net.synapses()), gbrain::Network::backend(),
              cfg.threads ? cfg.threads : 0u, buildSeconds);

  // A stimulus: a random tenth of the excitatory neurons get a constant current.
  std::vector<float> I(cfg.neurons, 0.0f);
  uint64_t r = 12345;
  for (uint32_t i = 0; i < cfg.neurons; i++) {
    r = r * 6364136223846793005ULL + 1442695040888963407ULL;
    if (!net.isInhibitory(i) && (r >> 33) % 10 == 0) I[i] = drive;
  }

  uint64_t spikes = 0;
  auto t2 = std::chrono::steady_clock::now();
  for (uint32_t t = 0; t < ticks; t++) spikes += net.step(I.data(), 1.0f).fired;
  auto t3 = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(t3 - t2).count();
  const double ticksPerSecond = ticks / seconds;
  const double rate = static_cast<double>(spikes) / ticks / cfg.neurons;
  std::printf("  %u ticks in %.3f s → %.1f ticks/s (%.2f ms/tick), %.2f M spikes/s, mean firing %.3f per neuron per tick\n",
              ticks, seconds, ticksPerSecond, 1000.0 / ticksPerSecond, spikes / seconds / 1e6, rate);
  std::printf("  real time at 10 Hz: %s (%.1f× real time at dt=1 ms as 1 tick)\n",
              ticksPerSecond >= 10.0 ? "yes" : "NO", ticksPerSecond / 10.0);
  return 0;
}
