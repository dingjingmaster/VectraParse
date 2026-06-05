#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THRDP_DIR="$PROJECT_ROOT/3thrd"
ORT_SRC="$THRDP_DIR/onnxruntime"
INSTALL_DIR="$SCRIPT_DIR/install"
STATIC_MODE=false
OPENMP=true
OPENMP_RUNTIME_DIR=""

for arg in "$@"; do
    case "$arg" in
        --static) STATIC_MODE=true ;;
        --no-openmp) OPENMP=false ;;
        --openmp) OPENMP=true ;;
        --openmp-runtime-dir=*)
            OPENMP_RUNTIME_DIR="${arg#*=}"
            ;;
        --help)
            echo "Usage: $0 [--static] [--openmp] [--no-openmp] [--openmp-runtime-dir=<dir>]"
            echo
            echo "  --static                 Build ONNX Runtime as static library"
            echo "  --openmp                 Enable ONNX Runtime OpenMP support (default)"
            echo "  --no-openmp              Disable ONNX Runtime OpenMP support"
            echo "  --openmp-runtime-dir=DIR  Optional path containing OpenMP runtime libs"
            echo "                           (for packaging static/shared ORT output)"
            exit 0
            ;;
    esac
done

if [ -n "$OPENMP_RUNTIME_DIR" ] && [ ! -d "$OPENMP_RUNTIME_DIR" ]; then
    echo "OpenMP runtime dir not found: $OPENMP_RUNTIME_DIR" >&2
    exit 1
fi

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

CMAKE_C_FLAGS="-fPIC"
CMAKE_CXX_FLAGS="-march=native -O3 -static-libgcc -static-libstdc++ -Wno-error=maybe-uninitialized"
OPENMP_CMAKE_OPTIONS=()
# Suppress FetchContent_Populate deprecation in CMake 3.24+ (e.g., ONNX Runtime's safeint fetch path).
# This keeps build output clean without modifying third-party CMake files.
ORT_CMAKE_POLICIES=(
    "-DCMAKE_POLICY_DEFAULT_CMP0169=OLD"
)

if $OPENMP; then
    echo "OpenMP: enabled"
    if grep -q "option(onnxruntime_USE_OPENMP" "$ORT_SRC/cmake/CMakeLists.txt" 2>/dev/null; then
        OPENMP_CMAKE_OPTIONS+=("-Donnxruntime_USE_OPENMP=ON")
    fi
fi

if $OPENMP; then
    CMAKE_C_FLAGS="$CMAKE_C_FLAGS -fopenmp"
    CMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS -fopenmp"
fi

find_openmp_runtime() {
    local names=(
        "libgomp.so"
        "libgomp.so.1"
        "libgomp.a"
        "libiomp5.so"
        "libiomp5.so.5"
        "libiomp5.a"
    )
    local compiler="${CC:-gcc}"
    local paths=()

    for n in "${names[@]}"; do
        if [ -n "$OPENMP_RUNTIME_DIR" ] && [ -f "$OPENMP_RUNTIME_DIR/$n" ]; then
            paths+=("$OPENMP_RUNTIME_DIR/$n")
            continue
        fi
        local lib_path
        lib_path="$("$compiler" -print-file-name="$n" 2>/dev/null || true)"
        if [ "$lib_path" != "$n" ] && [ -f "$lib_path" ]; then
            paths+=("$lib_path")
        fi
    done

    printf '%s\n' "${paths[@]}" | awk 'NF && !seen[$0]++'
}

OPENMP_RUNTIME_LIBS=()
if $OPENMP; then
    while IFS= read -r lib; do
        OPENMP_RUNTIME_LIBS+=("$lib")
    done < <(find_openmp_runtime)
fi

if [ "${#OPENMP_RUNTIME_LIBS[@]}" -gt 0 ]; then
    echo "OpenMP runtime libs:"
    printf '  %s\n' "${OPENMP_RUNTIME_LIBS[@]}"
fi

OPENMP_STATIC_ARCHIVES=()
OPENMP_SHARED_LIBS=()
if $OPENMP; then
    for lib in "${OPENMP_RUNTIME_LIBS[@]}"; do
        case "$lib" in
            *.a)
                OPENMP_STATIC_ARCHIVES+=("$lib")
                ;;
            *.so | *.so.*)
                OPENMP_SHARED_LIBS+=("$lib")
                ;;
        esac
    done
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
    "${ORT_CMAKE_POLICIES[@]}" \
    "${OPENMP_CMAKE_OPTIONS[@]}" \
    -DCMAKE_C_FLAGS="$CMAKE_C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" \
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

if [ "${#OPENMP_SHARED_LIBS[@]}" -gt 0 ]; then
    echo ""
    echo "=== Packaging OpenMP shared runtime libraries ==="
    mkdir -p "$INSTALL_DIR/lib"
    for lib in "${OPENMP_SHARED_LIBS[@]}"; do
        cp -a "$lib" "$INSTALL_DIR/lib/"
    done
    ls -la "$INSTALL_DIR/lib/" | grep -E "libgomp|libiomp5" || true
fi

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

    mapfile -t archive_list < <({
        find "$BUILD_DIR" -name "*.a" ! -path "$STATIC_DIR/*"
        printf '%s\n' "${OPENMP_STATIC_ARCHIVES[@]}"
    } | sort -u)
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

    if [ "${#OPENMP_STATIC_ARCHIVES[@]}" -gt 0 ]; then
        echo "Included OpenMP static archives:"
        printf '  %s\n' "${OPENMP_STATIC_ARCHIVES[@]}"
    fi

    echo "Static library: $STATIC_LIB_DIR/libonnxruntime_all.a"
    ls -lh "$STATIC_LIB_DIR/libonnxruntime_all.a"
fi

echo ""
echo "=== Done ==="
echo "Library: $INSTALL_DIR/lib/"
echo "Headers: $INSTALL_DIR/include/"
ls -la "$INSTALL_DIR/lib/"*onnxruntime* 2>/dev/null || true
