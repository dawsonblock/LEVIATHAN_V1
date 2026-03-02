#!/bin/bash
# build_h100.sh - Production build script for H100 native compilation

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "=========================================="
echo "Leviathan CSR Apex v3.2 - H100 Build"
echo "=========================================="
echo ""
echo "CUDA Path:      $CUDA_PATH"
echo "Python Prefix:  $PYTHON_PREFIX"
echo "Site Packages:  $SITE_PACKAGES"
echo "Build Threads:  $NPROC"
echo ""

mkdir -p build
cd build

echo "[1/3] Running CMake configuration..."
cmake .. \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo "[2/3] Compiling CUDA kernels and Pybind11 module..."
make -j"$NPROC"

echo ""
echo "[3/3] Installing Python module..."
SOFILE=$(find . -name 'leviathan_cuda*.so' | head -1)
if [ -z "$SOFILE" ]; then
    echo "ERROR: leviathan_cuda.so not found!"
    exit 1
fi

cp "$SOFILE" "$SITE_PACKAGES/" 2>/dev/null || cp "$SOFILE" .
echo "  Installed to: $SITE_PACKAGES/ (or local fallback)"

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Module: $(find . -name 'leviathan_cuda*.so' | head -1)"
echo "Ready for deployment on H100"
echo ""
echo "Next: python3 leviathan_h100.py"
echo ""