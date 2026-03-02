#!/bin/bash
# build_h100.sh — Production build for Leviathan v3.3

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "================================================"
echo "  LEVIATHAN CSR Apex v3.3 — H100 Build"
echo "================================================"
echo ""
echo "  CUDA:    $CUDA_PATH"
echo "  Python:  $PYTHON_PREFIX"
echo "  Threads: $NPROC"
echo ""

# Build from project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

mkdir -p build && cd build

echo "[1/3] CMake configuration..."
cmake .. \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[2/3] Compiling CUDA kernels + Pybind11 modules..."
make -j"$NPROC"

echo ""
echo "[3/3] Installing Python modules..."
for SOFILE in $(find . -name 'leviathan_*.so'); do
    BASENAME=$(basename "$SOFILE")
    cp "$SOFILE" "$SITE_PACKAGES/" 2>/dev/null || cp "$SOFILE" .
    echo "  → $BASENAME"
done

echo ""
echo "================================================"
echo "  Build Complete!"
echo "================================================"
echo ""
echo "  Run:  python3 python/leviathan_h100.py"
echo "  Dash: python3 python/dashboard.py"
echo ""