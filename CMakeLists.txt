cmake_minimum_required(VERSION 3.14)
project(tai-compressor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# ── balanced ──────────────────────────────────────────────────────────────────
file(GLOB BALANCED_SOURCES
    src/balanced/coder/*.cpp
)

add_executable(compress_balanced
    src/balanced/main_compress.cpp ${BALANCED_SOURCES})
target_include_directories(compress_balanced PRIVATE src/balanced)
target_link_libraries(compress_balanced PRIVATE Threads::Threads)

add_executable(decompress_balanced
    src/balanced/main_decompress.cpp ${BALANCED_SOURCES})
target_include_directories(decompress_balanced PRIVATE src/balanced)
target_link_libraries(decompress_balanced PRIVATE Threads::Threads)

# ── fast ──────────────────────────────────────────────────────────────────────
file(GLOB FAST_SOURCES
    src/fast/coder/*.cpp
)

add_executable(compress_fast
    src/fast/main_compress.cpp ${FAST_SOURCES})
target_include_directories(compress_fast PRIVATE src/fast)
target_link_libraries(compress_fast PRIVATE Threads::Threads)

add_executable(decompress_fast
    src/fast/main_decompress.cpp ${FAST_SOURCES})
target_include_directories(decompress_fast PRIVATE src/fast)
target_link_libraries(decompress_fast PRIVATE Threads::Threads)

# ── ratio ─────────────────────────────────────────────────────────────────────
file(GLOB RATIO_SOURCES
    src/ratio/coder/*.cpp
    src/ratio/model/*.cpp
)

add_executable(compress_ratio
    src/ratio/main_compress.cpp ${RATIO_SOURCES})
target_include_directories(compress_ratio PRIVATE src/ratio)
target_link_libraries(compress_ratio PRIVATE Threads::Threads)

add_executable(decompress_ratio
    src/ratio/main_decompress.cpp ${RATIO_SOURCES})
target_include_directories(decompress_ratio PRIVATE src/ratio)
target_link_libraries(decompress_ratio PRIVATE Threads::Threads)

# ── astro ─────────────────────────────────────────────────────────────────────
file(GLOB ASTRO_SOURCES
    src/astro/coder/*.cpp
)

add_executable(compress_astro
    src/astro/main_compress.cpp ${ASTRO_SOURCES})
target_include_directories(compress_astro PRIVATE src/astro)

add_executable(decompress_astro
    src/astro/main_decompress.cpp ${ASTRO_SOURCES})
target_include_directories(decompress_astro PRIVATE src/astro)
