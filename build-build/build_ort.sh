#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THRDP_DIR="$PROJECT_ROOT/3thrd"
ORT_SRC="$THRDP_DIR/onnxruntime"
INSTALL_DIR="$SCRIPT_DIR/install"
STATIC_MODE=false

for arg in "$@"; do
    case "$arg" in
        --static) STATIC_MODE=true ;;
    esac
done

BUILD_DIR="$SCRIPT_DIR/ort_build"
if $STATIC_MODE; then
    BUILD_DIR="$SCRIPT_DIR/ort_build_static"
fi

echo "=== Building ONNX Runtime v$(cat "$ORT_SRC/VERSION_NUMBER") (static=$STATIC_MODE) ==="
echo "Project root: $PROJECT_ROOT"
echo "Build dir:    $BUILD_DIR"
echo "Install dir:  $INSTALL_DIR"

mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

LOCAL_DEPS=(
    abseil_cpp
    Protobuf
    re2
    Eigen3
    date
    nlohmann_json
    GSL
    safeint
    flatbuffers
    mp11
    pytorch_cpuinfo
    onnx
)

CMAKE_DEPS_FLAGS=()
for dep in "${LOCAL_DEPS[@]}"; do
    case "$dep" in
        abseil_cpp)   local_dir="$THRDP_DIR/abseil-cpp" ;;
        Protobuf)     local_dir="$THRDP_DIR/protobuf" ;;
        re2)          local_dir="$THRDP_DIR/re2" ;;
        Eigen3)       local_dir="$THRDP_DIR/eigen" ;;
        date)         local_dir="$THRDP_DIR/date" ;;
        nlohmann_json) local_dir="$THRDP_DIR/json" ;;
        GSL)          local_dir="$THRDP_DIR/GSL" ;;
        safeint)      local_dir="$THRDP_DIR/safeInt" ;;
        flatbuffers)  local_dir="$THRDP_DIR/flatbuffers" ;;
        mp11)         local_dir="/usr" ;;
        pytorch_cpuinfo) local_dir="$THRDP_DIR/cpuinfo" ;;
        onnx)         local_dir="$THRDP_DIR/onnx" ;;
    esac
    if [ -d "$local_dir" ]; then
        dep_var_name="${dep^^}"
        CMAKE_DEPS_FLAGS+=("-DFETCHCONTENT_SOURCE_DIR_${dep_var_name}=${local_dir}")
        echo "  Using local dep: $dep (${dep_var_name}) -> $local_dir"
    fi
done

echo ""
echo "=== Configuring with CMake ==="
SHARED_LIB_FLAG=ON
if $STATIC_MODE; then
    SHARED_LIB_FLAG=OFF
fi

cmake -S "$ORT_SRC/cmake" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -Donnxruntime_BUILD_SHARED_LIB="$SHARED_LIB_FLAG" \
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
    -DCMAKE_CXX_FLAGS="-march=native -O3 -static-libgcc -static-libstdc++ -Wno-error=maybe-uninitialized" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

echo ""
echo "=== Building ==="
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"
if $STATIC_MODE; then
    cmake --build "$BUILD_DIR" --config Release --target re2 -j"$(nproc)"
fi

echo ""
echo "=== Installing ==="
cmake --install "$BUILD_DIR" --config Release --prefix "$INSTALL_DIR"

if $STATIC_MODE; then
    echo ""
    echo "=== Merging static libraries ==="
    STATIC_DIR="$INSTALL_DIR/static"
    STATIC_LIB_DIR="$STATIC_DIR/lib"
    mkdir -p "$STATIC_LIB_DIR"
    rm -f "$STATIC_LIB_DIR/libonnxruntime_all.a"
    # Copy headers from install
    rm -rf "$STATIC_DIR/include"
    cp -r "$INSTALL_DIR/include" "$STATIC_DIR/"

    mapfile -t archive_list < <(find "$BUILD_DIR" -name "*.a" ! -path "$STATIC_DIR/*" | sort)
    if [ "${#archive_list[@]}" -eq 0 ]; then
        echo "no static archives found under $BUILD_DIR" >&2
        exit 1
    fi

    MRI_SCRIPT="$(mktemp)"
    cleanup_mri() {
        rm -f "$MRI_SCRIPT"
    }
    trap cleanup_mri EXIT
    {
        printf 'CREATE %s\n' "$STATIC_LIB_DIR/libonnxruntime_all.a"
        for archive in "${archive_list[@]}"; do
            printf 'ADDLIB %s\n' "$archive"
        done
        printf 'SAVE\n'
        printf 'END\n'
    } > "$MRI_SCRIPT"
    ar -M < "$MRI_SCRIPT"
    ranlib "$STATIC_LIB_DIR/libonnxruntime_all.a"

    echo "Static library: $STATIC_LIB_DIR/libonnxruntime_all.a"
    ls -lh "$STATIC_LIB_DIR/libonnxruntime_all.a"
fi

echo ""
echo "=== Done ==="
echo "Library: $INSTALL_DIR/lib/"
echo "Headers: $INSTALL_DIR/include/"
ls -la "$INSTALL_DIR/lib/"*onnxruntime* 2>/dev/null || true
