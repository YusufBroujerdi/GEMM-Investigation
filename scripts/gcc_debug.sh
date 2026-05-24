#!/usr/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -z "${CONDA_PREFIX}" ]; then
    echo "Error: no active conda environment found."
    echo "This script is intended to be used with an activated conda env"
    echo "Use the yaml at the root to create a conda env"
    exit 1
fi

"$CONDA_PREFIX/bin/python" -c "import numpy; print(numpy.__version__)"

cmake -S . -B build-gcc-debug \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_EXPORT_COMPILER_COMMANDS=ON \
-DPython3_EXECUTABLE="${CONDA_PREFIX}/bin/python"

cmake --build build-gcc-debug

ctest --test-dir build-gcc-debug --output-on-failure
