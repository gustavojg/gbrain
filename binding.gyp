{
  "targets": [
    {
      "target_name": "gbrain_native",
      "sources": ["native/addon.cc", "native/src/engine.cpp"],
      "include_dirs": ["<!@(node -p \"require('node-addon-api').include\")", "native"],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS", "NAPI_VERSION=8"],
      "cflags_cc": ["-std=c++17", "-O3"],
      "xcode_settings": {
        "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
        "GCC_OPTIMIZATION_LEVEL": "3",
        "MACOSX_DEPLOYMENT_TARGET": "12.0"
      },
      "msvs_settings": { "VCCLCompilerTool": { "AdditionalOptions": ["/std:c++17", "/O2"] } }
    }
  ]
}
