#!/bin/bash
# build_h100.sh — Build for SM86 (3080 Ti) / SM90 (H100)

set -e

CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "================================================"
echo "  LEVIATHAN — Build"
echo "================================================"
echo "  CUDA:    $CUDA_PATH"
echo "  Threads: $NPROC"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

rm -rf build && mkdir build && cd build

echo "[1/3] CMake..."
cmake .. \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    -DCMAKE_BUILD_TYPE=Release

echo "[2/3] Compiling..."
make -j"$NPROC"

echo "[3/3] Installing modules..."
for SOFILE in $(find . -name 'leviathan_*.so'); do
    BASENAME=$(basename "$SOFILE")
    cp "$SOFILE" "$SITE_PACKAGES/" 2>/dev/null || cp "$SOFILE" "$PROJECT_ROOT/"
    echo "  → $BASENAME"
done

echo ""
echo "================================================"
echo "  Done. Verify:"
echo "    python3 -c \"from leviathan_cuda import LeviathanEngine\""
echo "    python3 -c \"from leviathan_phi import GPUPhiWorker\""
echo "================================================"