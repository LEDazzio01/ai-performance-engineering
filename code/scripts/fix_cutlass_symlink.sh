#!/usr/bin/env bash
# Auto-repair script for CUTLASS symlink issues
# Run this if you see "missing header: tmem_allocator_sm100.hpp" errors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUTLASS_SRC="${PROJECT_ROOT}/third_party/cutlass"
TE_CUTLASS="${PROJECT_ROOT}/third_party/TransformerEngine/3rdparty/cutlass"

echo "🔧 CUTLASS Symlink Auto-Repair"
echo "=============================="
echo ""

# Check if main CUTLASS exists
if [ ! -d "${CUTLASS_SRC}" ]; then
    echo "❌ Main CUTLASS not found at: ${CUTLASS_SRC}"
    echo "   Run: ./scripts/install_cutlass.sh"
    exit 1
fi

# Check CUTLASS version
CUTLASS_MAJOR=$(grep -E "^#define CUTLASS_MAJOR" "${CUTLASS_SRC}/include/cutlass/version.h" | awk '{print $3}')
CUTLASS_MINOR=$(grep -E "^#define CUTLASS_MINOR" "${CUTLASS_SRC}/include/cutlass/version.h" | awk '{print $3}')
CUTLASS_PATCH=$(grep -E "^#define CUTLASS_PATCH" "${CUTLASS_SRC}/include/cutlass/version.h" | awk '{print $3}')
CUTLASS_VERSION="${CUTLASS_MAJOR}.${CUTLASS_MINOR}.${CUTLASS_PATCH}"
echo "✓ Main CUTLASS version: ${CUTLASS_VERSION}"

# Check SM100 headers
if [ ! -f "${CUTLASS_SRC}/include/cute/arch/tmem_allocator_sm100.hpp" ]; then
    echo "❌ Main CUTLASS missing SM100a headers (version too old)"
    echo "   Run: ./scripts/install_cutlass.sh"
    exit 1
fi
echo "✓ SM100a headers present"

# Check TransformerEngine exists
if [ ! -d "${PROJECT_ROOT}/third_party/TransformerEngine" ]; then
    echo "ℹ️  TransformerEngine not installed - nothing to fix"
    exit 0
fi

# Check current state of TE's CUTLASS
echo ""
echo "Checking TransformerEngine CUTLASS..."

if [ -L "${TE_CUTLASS}" ]; then
    TARGET=$(readlink -f "${TE_CUTLASS}" 2>/dev/null || true)
    EXPECTED=$(readlink -f "${CUTLASS_SRC}" 2>/dev/null || true)
    
    if [ "${TARGET}" = "${EXPECTED}" ]; then
        echo "✓ TE CUTLASS symlink already correct"
        echo "   Points to: ${TARGET}"
        exit 0
    else
        echo "⚠️  TE CUTLASS symlink points to wrong target"
        echo "   Current: ${TARGET}"
        echo "   Expected: ${EXPECTED}"
    fi
elif [ -d "${TE_CUTLASS}" ]; then
    echo "⚠️  TE CUTLASS is a directory (not a symlink)"
    # Check its version
    if [ -f "${TE_CUTLASS}/include/cutlass/version.h" ]; then
        TE_CUTLASS_VER=$(grep -E "CUTLASS_(MAJOR|MINOR|PATCH)" "${TE_CUTLASS}/include/cutlass/version.h" | \
            sed -E 's/.*([0-9]+)/\1/' | tr '\n' '.' | sed 's/\.$//')
        echo "   Version: ${TE_CUTLASS_VER} (likely outdated)"
    fi
else
    echo "⚠️  TE CUTLASS missing entirely"
fi

# Fix it
echo ""
echo "Fixing TE CUTLASS symlink..."
rm -rf "${TE_CUTLASS}"
ln -s "${CUTLASS_SRC}" "${TE_CUTLASS}"

# Verify fix
if [ -L "${TE_CUTLASS}" ] && [ -f "${TE_CUTLASS}/include/cute/arch/tmem_allocator_sm100.hpp" ]; then
    echo ""
    echo "✅ Fixed! TE CUTLASS now symlinked to ${CUTLASS_VERSION}"
    echo "   ${TE_CUTLASS} -> ${CUTLASS_SRC}"
else
    echo ""
    echo "❌ Fix failed - please check manually"
    exit 1
fi

