#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THRDP_DIR="$PROJECT_ROOT/3thrd"
ORT_SRC="$THRDP_DIR/onnxruntime"
BUILD_DIR="$SCRIPT_DIR/ort_build"
INSTALL_DIR="$SCRIPT_DIR/install"

echo "=== Building ONNX Runtime v$(cat "$ORT_SRC/VERSION_NUMBER") ==="
echo "Project root: $PROJECT_ROOT"
echo "Build dir:    $BUILD_DIR"
echo "Install dir:  $INSTALL_DIR"

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

LOCAL_DEPS=(
    abseil_cpp
    Protobuf
    re2
    Eigen3
    nlohmann_json
    GSL
    safeint
    flatbuffers
    onnx
)

CMAKE_DEPS_FLAGS=()
for dep in "${LOCAL_DEPS[@]}"; do
    case "$dep" in
        abseil_cpp)   local_dir="$THRDP_DIR/abseil-cpp" ;;
        Protobuf)     local_dir="$THRDP_DIR/protobuf" ;;
        re2)          local_dir="$THRDP_DIR/re2" ;;
        Eigen3)       local_dir="$THRDP_DIR/eigen" ;;
        nlohmann_json) local_dir="$THRDP_DIR/json" ;;
        GSL)          local_dir="$THRDP_DIR/GSL" ;;
        safeint)      local_dir="$THRDP_DIR/safeInt" ;;
        flatbuffers)  local_dir="$THRDP_DIR/flatbuffers" ;;
        onnx)         local_dir="$THRDP_DIR/onnx" ;;
    esac
    if [ -d "$local_dir" ]; then
        CMAKE_DEPS_FLAGS+=("-DFETCHCONTENT_SOURCE_DIR_${dep}=${local_dir}")
        echo "  Using local dep: $dep -> $local_dir"
    fi
done

echo ""
echo "=== Configuring with CMake ==="
cmake -S "$ORT_SRC/cmake" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -Donnxruntime_BUILD_SHARED_LIB=ON \
    -Donnxruntime_BUILD_UNIT_TESTS=OFF \
    -Donnxruntime_ENABLE_PYTHON=OFF \
    -Donnxruntime_ENABLE_TRAINING=OFF \
    -Donnxruntime_USE_CUDA=OFF \
    -Donnxruntime_USE_DNNL=OFF \
    -Donnxruntime_USE_TENSORRT=OFF \
    -Donnxruntime_USE_OPENVINO=OFF \
    -Donnxruntime_USE_COREML=OFF \
    -Donnxruntime_USE_XNNPACK=OFF \
    -Donnxruntime_USE_WEBGPU=OFF \
    -Donnxruntime_DISABLE_CONTRIB_OPS=ON \
    -Donnxruntime_DISABLE_ML_OPS=ON \
    -Donnxruntime_DISABLE_GENERATION_OPS=ON \
    -Donnxruntime_ENABLE_CPU_FP16_OPS=OFF \
    -Donnxruntime_USE_FULL_PROTOBUF=OFF \
    -Donnxruntime_ENABLE_CPUINFO=ON \
    "${CMAKE_DEPS_FLAGS[@]}" \
    -DCMAKE_CXX_FLAGS="-march=native -O3" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo ""
echo "=== Building ==="
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

echo ""
echo "=== Installing ==="
cmake --install "$BUILD_DIR" --config Release --prefix "$INSTALL_DIR"

echo ""
echo "=== Done ==="
echo "Library: $INSTALL_DIR/lib/"
echo "Headers: $INSTALL_DIR/include/"
ls -la "$INSTALL_DIR/lib/"*onnxruntime* 2>/dev/null || true
