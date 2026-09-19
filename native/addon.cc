// gBrain native engine — Node addon (N-API)
// ============================================================================
// Exposes the engine to the brain:
//
//   const { NativeNetwork, backend } = require('gbrain_native.node');
//   const net = new NativeNetwork({ neurons, fanIn, inhibitoryFraction, seed, threads, plastic });
//   const fired = net.step(externalCurrent /* Float32Array | null */, modulation) // → Uint32Array
//   net.neurons, net.synapses, net.potentials() → Float32Array, net.weights() → Float32Array
//   net.setWeights(Float32Array), net.setSynapses(rowPtr, targets, weights), net.reset()
#include <napi.h>

#include <memory>

#include "src/engine.h"

namespace {

class NativeNetwork : public Napi::ObjectWrap<NativeNetwork> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "NativeNetwork", {
      InstanceMethod("step", &NativeNetwork::Step),
      InstanceMethod("stepChannels", &NativeNetwork::StepChannels),
      InstanceMethod("setInputProjection", &NativeNetwork::SetInputProjection),
      InstanceMethod("buildRandomInputProjection", &NativeNetwork::BuildRandomInputProjection),
      InstanceMethod("potentials", &NativeNetwork::Potentials),
      InstanceMethod("weights", &NativeNetwork::Weights),
      InstanceMethod("synapseRowPtr", &NativeNetwork::SynapseRowPtr),
      InstanceMethod("synapseTargets", &NativeNetwork::SynapseTargets),
      InstanceMethod("setWeights", &NativeNetwork::SetWeights),
      InstanceMethod("setSynapses", &NativeNetwork::SetSynapses),
      InstanceMethod("isInhibitory", &NativeNetwork::IsInhibitory),
      InstanceMethod("reset", &NativeNetwork::Reset),
      InstanceAccessor("neurons", &NativeNetwork::Neurons, nullptr),
      InstanceAccessor("synapses", &NativeNetwork::Synapses, nullptr),
    });
    exports.Set("NativeNetwork", func);
    exports.Set("backend", Napi::String::New(env, gbrain::Network::backend()));
    return exports;
  }

  explicit NativeNetwork(const Napi::CallbackInfo& info) : Napi::ObjectWrap<NativeNetwork>(info) {
    Napi::Env env = info.Env();
    gbrain::NetworkConfig cfg;
    if (info.Length() > 0 && info[0].IsObject()) {
      Napi::Object o = info[0].As<Napi::Object>();
      auto num = [&](const char* key, double fallback) -> double {
        Napi::Value v = o.Get(key);
        return v.IsNumber() ? v.As<Napi::Number>().DoubleValue() : fallback;
      };
      cfg.neurons = static_cast<uint32_t>(num("neurons", cfg.neurons));
      cfg.fanIn = static_cast<uint32_t>(num("fanIn", cfg.fanIn));
      cfg.inhibitoryFraction = static_cast<float>(num("inhibitoryFraction", cfg.inhibitoryFraction));
      cfg.excMin = static_cast<float>(num("excMin", cfg.excMin));
      cfg.excMax = static_cast<float>(num("excMax", cfg.excMax));
      cfg.inhWeight = static_cast<float>(num("inhWeight", cfg.inhWeight));
      cfg.excToInhGain = static_cast<float>(num("excToInhGain", cfg.excToInhGain));
      cfg.dt = static_cast<float>(num("dt", cfg.dt));
      cfg.noise = static_cast<float>(num("noise", cfg.noise));
      cfg.aPlus = static_cast<float>(num("aPlus", cfg.aPlus));
      cfg.aMinus = static_cast<float>(num("aMinus", cfg.aMinus));
      cfg.tauPlus = static_cast<float>(num("tauPlus", cfg.tauPlus));
      cfg.tauMinus = static_cast<float>(num("tauMinus", cfg.tauMinus));
      cfg.wMax = static_cast<float>(num("wMax", cfg.wMax));
      cfg.threads = static_cast<uint32_t>(num("threads", cfg.threads));
      cfg.seed = static_cast<uint64_t>(num("seed", static_cast<double>(cfg.seed)));
      Napi::Value plastic = o.Get("plastic");
      if (plastic.IsBoolean()) cfg.plastic = plastic.As<Napi::Boolean>().Value();
    }
    if (cfg.neurons == 0 || cfg.neurons > 50000000u) {
      Napi::RangeError::New(env, "neurons must be in [1, 50 000 000]").ThrowAsJavaScriptException();
      return;
    }
    net_ = std::make_unique<gbrain::Network>(cfg);
  }

 private:
  Napi::Value Step(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const float* external = nullptr;
    if (info.Length() > 0 && info[0].IsTypedArray()) {
      Napi::Float32Array arr = info[0].As<Napi::Float32Array>();
      if (arr.ElementLength() != net_->neurons()) {
        Napi::RangeError::New(env, "externalCurrent must have one entry per neuron").ThrowAsJavaScriptException();
        return env.Undefined();
      }
      external = arr.Data();
    }
    const float modulation = info.Length() > 1 && info[1].IsNumber() ? info[1].As<Napi::Number>().FloatValue() : 1.0f;
    net_->step(external, modulation);
    const auto& fired = net_->fired();
    Napi::Uint32Array out = Napi::Uint32Array::New(env, fired.size());
    for (size_t i = 0; i < fired.size(); i++) out[i] = fired[i];
    return out;
  }

  Napi::Value StepChannels(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
      Napi::TypeError::New(env, "stepChannels(channels: Float32Array, modulation?)").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    Napi::Float32Array arr = info[0].As<Napi::Float32Array>();
    if (arr.ElementLength() != net_->inputChannels()) {
      Napi::RangeError::New(env, "channels must have one entry per input channel of the projection").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    const float modulation = info.Length() > 1 && info[1].IsNumber() ? info[1].As<Napi::Number>().FloatValue() : 1.0f;
    net_->stepChannels(arr.Data(), modulation);
    const auto& fired = net_->fired();
    Napi::Uint32Array out = Napi::Uint32Array::New(env, fired.size());
    for (size_t i = 0; i < fired.size(); i++) out[i] = fired[i];
    return out;
  }

  Napi::Value SetInputProjection(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4 || !info[0].IsNumber() || !info[1].IsTypedArray() || !info[2].IsTypedArray() || !info[3].IsTypedArray()) {
      Napi::TypeError::New(env, "setInputProjection(channels, rowPtr: Uint32Array, cols: Uint32Array, weights: Float32Array)").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    const uint32_t channels = info[0].As<Napi::Number>().Uint32Value();
    Napi::Uint32Array rowPtr = info[1].As<Napi::Uint32Array>();
    Napi::Uint32Array cols = info[2].As<Napi::Uint32Array>();
    Napi::Float32Array weights = info[3].As<Napi::Float32Array>();
    const uint32_t n = net_->neurons();
    if (rowPtr.ElementLength() != n + 1 || cols.ElementLength() != weights.ElementLength() || rowPtr[n] != cols.ElementLength()) {
      Napi::RangeError::New(env, "rowPtr must have neurons+1 entries and end at the synapse count").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    for (size_t i = 0; i < cols.ElementLength(); i++) if (cols[i] >= channels) {
      Napi::RangeError::New(env, "a channel index is out of range").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    net_->setInputProjection(channels, std::vector<uint32_t>(rowPtr.Data(), rowPtr.Data() + rowPtr.ElementLength()),
                             std::vector<uint32_t>(cols.Data(), cols.Data() + cols.ElementLength()),
                             std::vector<float>(weights.Data(), weights.Data() + weights.ElementLength()));
    return env.Undefined();
  }

  Napi::Value BuildRandomInputProjection(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsNumber()) {
      Napi::TypeError::New(env, "buildRandomInputProjection(channels, fanIn, wMin?, wMax?)").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    const float wMin = info.Length() > 2 && info[2].IsNumber() ? info[2].As<Napi::Number>().FloatValue() : 0.0f;
    const float wMax = info.Length() > 3 && info[3].IsNumber() ? info[3].As<Napi::Number>().FloatValue() : 1.0f;
    net_->buildRandomInputProjection(info[0].As<Napi::Number>().Uint32Value(), info[1].As<Napi::Number>().Uint32Value(), wMin, wMax);
    return env.Undefined();
  }

  Napi::Value Potentials(const Napi::CallbackInfo& info) {
    const auto& v = net_->potentials();
    Napi::Float32Array out = Napi::Float32Array::New(info.Env(), v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = v[i];
    return out;
  }

  Napi::Value Weights(const Napi::CallbackInfo& info) {
    const auto& w = net_->weights();
    Napi::Float32Array out = Napi::Float32Array::New(info.Env(), w.size());
    for (size_t i = 0; i < w.size(); i++) out[i] = w[i];
    return out;
  }

  Napi::Value SynapseRowPtr(const Napi::CallbackInfo& info) {
    const auto& r = net_->rowPtr();
    Napi::Uint32Array out = Napi::Uint32Array::New(info.Env(), r.size());
    for (size_t i = 0; i < r.size(); i++) out[i] = r[i];
    return out;
  }

  Napi::Value SynapseTargets(const Napi::CallbackInfo& info) {
    const auto& t = net_->targets();
    Napi::Uint32Array out = Napi::Uint32Array::New(info.Env(), t.size());
    for (size_t i = 0; i < t.size(); i++) out[i] = t[i];
    return out;
  }

  Napi::Value SetWeights(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsTypedArray()) {
      Napi::TypeError::New(env, "setWeights(Float32Array)").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    Napi::Float32Array arr = info[0].As<Napi::Float32Array>();
    auto& w = net_->weights();
    if (arr.ElementLength() != w.size()) {
      Napi::RangeError::New(env, "weights must have one entry per synapse").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    for (size_t i = 0; i < w.size(); i++) w[i] = arr[i];
    return env.Undefined();
  }

  Napi::Value SetSynapses(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsTypedArray() || !info[1].IsTypedArray() || !info[2].IsTypedArray()) {
      Napi::TypeError::New(env, "setSynapses(rowPtr: Uint32Array, targets: Uint32Array, weights: Float32Array)").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    Napi::Uint32Array rowPtr = info[0].As<Napi::Uint32Array>();
    Napi::Uint32Array targets = info[1].As<Napi::Uint32Array>();
    Napi::Float32Array weights = info[2].As<Napi::Float32Array>();
    const uint32_t n = net_->neurons();
    if (rowPtr.ElementLength() != n + 1 || targets.ElementLength() != weights.ElementLength() || rowPtr[n] != targets.ElementLength()) {
      Napi::RangeError::New(env, "rowPtr must have neurons+1 entries and end at the synapse count").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    std::vector<uint32_t> rp(rowPtr.Data(), rowPtr.Data() + rowPtr.ElementLength());
    std::vector<uint32_t> tg(targets.Data(), targets.Data() + targets.ElementLength());
    for (uint32_t t : tg) if (t >= n) {
      Napi::RangeError::New(env, "a target is out of range").ThrowAsJavaScriptException();
      return env.Undefined();
    }
    std::vector<float> ws(weights.Data(), weights.Data() + weights.ElementLength());
    net_->setSynapses(std::move(rp), std::move(tg), std::move(ws));
    return env.Undefined();
  }

  Napi::Value IsInhibitory(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    const uint32_t i = info.Length() > 0 && info[0].IsNumber() ? info[0].As<Napi::Number>().Uint32Value() : 0;
    return Napi::Boolean::New(env, i < net_->neurons() && net_->isInhibitory(i));
  }

  Napi::Value Reset(const Napi::CallbackInfo& info) {
    net_->resetState();
    return info.Env().Undefined();
  }

  Napi::Value Neurons(const Napi::CallbackInfo& info) { return Napi::Number::New(info.Env(), net_->neurons()); }
  Napi::Value Synapses(const Napi::CallbackInfo& info) { return Napi::Number::New(info.Env(), static_cast<double>(net_->synapses())); }

  std::unique_ptr<gbrain::Network> net_;
};

Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  return NativeNetwork::Init(env, exports);
}

}  // namespace

NODE_API_MODULE(gbrain_native, InitAll)
