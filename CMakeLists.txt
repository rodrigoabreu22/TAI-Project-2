cmake_minimum_required(VERSION 3.14)
project(tai-compressor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# ── astro-ratio ───────────────────────────────────────────────────────────────
# MED predictor + ORDER-1 hi (256 models) + ORDER-1(hi) lo (256 models)
# Best compression ratio. Rodrigo's work.
file(GLOB RATIO_SOURCES src/ratio/coder/*.cpp)

add_executable(compress_astro_ratio
    src/ratio/main_compress.cpp ${RATIO_SOURCES})
target_include_directories(compress_astro_ratio PRIVATE src/ratio)

add_executable(decompress_astro_ratio
    src/ratio/main_decompress.cpp ${RATIO_SOURCES})
target_include_directories(decompress_astro_ratio PRIVATE src/ratio)

# ── astro-balanced ────────────────────────────────────────────────────────────
# MED predictor + ORDER-1 hi (256 models) + ORDER-0 lo (1 model)
# Good ratio with simpler/faster coding stage. Hugo's work.
file(GLOB BALANCED_SOURCES src/balanced/coder/*.cpp)

add_executable(compress_astro_balanced
    src/balanced/main_compress.cpp ${BALANCED_SOURCES})
target_include_directories(compress_astro_balanced PRIVATE src/balanced)

add_executable(decompress_astro_balanced
    src/balanced/main_decompress.cpp ${BALANCED_SOURCES})
target_include_directories(decompress_astro_balanced PRIVATE src/balanced)

# ── astro-fast ────────────────────────────────────────────────────────────────
# Horizontal delta + ORDER-0 hi + ORDER-0 lo (2 models total)
# Minimum latency. Eduardo's work.
file(GLOB FAST_SOURCES src/fast/coder/*.cpp)

add_executable(compress_astro_fast
    src/fast/main_compress.cpp ${FAST_SOURCES})
target_include_directories(compress_astro_fast PRIVATE src/fast)

add_executable(decompress_astro_fast
    src/fast/main_decompress.cpp ${FAST_SOURCES})
target_include_directories(decompress_astro_fast PRIVATE src/fast)
