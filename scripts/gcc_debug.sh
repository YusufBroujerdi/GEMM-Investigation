#!/usr/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

cmake -S . -B build-gcc-debug \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_EXPORT_COMPILER_COMMANDS=ON

cmake --build build-gcc-debug

ctest --test-dir build-gcc-debug --output-on-failure
