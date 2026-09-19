cmake_minimum_required(VERSION 3.18)
project(gbrain_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GBRAIN_CUDA "Build the CUDA backend (needs the CUDA toolkit and an NVIDIA GPU)" OFF)

add_library(gbrain_engine STATIC src/engine.cpp)
target_include_directories(gbrain_engine PUBLIC src)
target_compile_options(gbrain_engine PRIVATE -O3)
find_package(Threads REQUIRED)
target_link_libraries(gbrain_engine PUBLIC Threads::Threads)

if(GBRAIN_CUDA)
  enable_language(CUDA)
  target_sources(gbrain_engine PRIVATE src/engine_cuda.cu)
  target_compile_definitions(gbrain_engine PUBLIC GBRAIN_CUDA)
  set_target_properties(gbrain_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(gbrain-bench src/bench.cpp)
target_link_libraries(gbrain-bench PRIVATE gbrain_engine)
