#!/bin/bash
# ECM Client Setup Script for Cloud Instances (vast.ai, etc.)
# Usage: curl -sSL https://ecm.kyleaskine.com/downloads/setup_cloud.sh | bash
#    or: wget -qO- https://ecm.kyleaskine.com/downloads/setup_cloud.sh | bash

set -e  # Exit on error

echo "============================================================"
echo "  ECM Factorization Client - Cloud Instance Setup"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Detect GPU Architecture (CUDA Compute Capability)
# ============================================================
echo "🔍 Detecting GPU architecture..."
ECM_VERSION="ecm86"  # Default fallback
CUDA_MAJOR=""

if command -v nvidia-smi &> /dev/null; then
    # Get CUDA compute capability and GPU name
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    COMPUTE_CAP_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs)

    if [ ! -z "$COMPUTE_CAP_RAW" ]; then
        # Remove decimal point (e.g., "8.6" -> "86")
        COMPUTE_CAP=$(echo "$COMPUTE_CAP_RAW" | tr -d '.' | tr -d ' ')

        echo "   GPU: $GPU_NAME (compute capability $COMPUTE_CAP_RAW)"

        # Map compute capability to ECM version
        # Note: Using string comparison for major version, then numeric for minor
        MAJOR=$(echo "$COMPUTE_CAP_RAW" | cut -d'.' -f1)
        MINOR=$(echo "$COMPUTE_CAP_RAW" | cut -d'.' -f2)

        if [ "$MAJOR" -ge "12" ]; then
            # Blackwell and newer (RTX 50-series, B-series, etc.)
            ECM_VERSION="ecm120"
            echo "✓ Using ecm120 (CUDA sm_120+ / Blackwell)"
        elif [ "$MAJOR" -ge "9" ]; then
            # Hopper and newer (H100, etc.)
            ECM_VERSION="ecm90"
            echo "✓ Using ecm90 (CUDA sm_90+ / Hopper)"
        elif [ "$MAJOR" -eq "8" ] && [ "$MINOR" -ge "6" ]; then
            # Ampere (RTX 3090, A100, etc.)
            ECM_VERSION="ecm86"
            echo "✓ Using ecm86 (CUDA sm_86 / Ampere)"
        elif [ "$MAJOR" -eq "7" ] && [ "$MINOR" -ge "5" ]; then
            # Turing (RTX 2080, T4, etc.)
            ECM_VERSION="ecm75"
            echo "✓ Using ecm75 (CUDA sm_75 / Turing)"
        else
            # Older GPUs - use ecm75 as fallback
            ECM_VERSION="ecm75"
            echo "✓ Using ecm75 (older GPU, compute $COMPUTE_CAP_RAW)"
        fi
    else
        echo "⚠️  Could not detect GPU compute capability, using default (ecm86)"
    fi
else
    echo "⚠️  nvidia-smi not found, using default (ecm86)"
fi

if command -v nvcc &> /dev/null; then
    NVCC_VERSION_OUTPUT=$(nvcc --version 2>/dev/null || true)
    CUDA_RELEASE=$(echo "$NVCC_VERSION_OUTPUT" | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)

    if [ ! -z "$CUDA_RELEASE" ]; then
        CUDA_MAJOR="$CUDA_RELEASE"
        echo "   CUDA toolkit: v$CUDA_MAJOR"

        if [ "$CUDA_MAJOR" -eq "13" ]; then
            case "$ECM_VERSION" in
                ecm86|ecm90|ecm120)
                    ECM_VERSION="${ECM_VERSION}v13"
                    echo "✓ Using CUDA v13 ECM binary directory: $ECM_VERSION"
                    ;;
                *)
                    echo "ℹ️  No CUDA v13-specific binary directory for $ECM_VERSION"
                    ;;
            esac
        fi
    else
        echo "⚠️  Could not parse nvcc version, using $ECM_VERSION"
    fi
else
    echo "ℹ️  nvcc not found, using $ECM_VERSION"
fi

echo "   Using ECM binary: $ECM_VERSION"

# ============================================================
# Step 2: Check/Install Dependencies
# ============================================================
echo ""
echo "📦 Checking dependencies..."

# Check if running as root or with sudo access
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Install git if needed
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y git
fi

# Install python3 and pip if needed
if ! command -v python3 &> /dev/null; then
    echo "Installing python3..."
    $SUDO apt-get install -y python3 python3-pip
fi

echo "✓ Dependencies ready"

# ============================================================
# Step 3: User Configuration
# ============================================================
echo ""
echo "============================================================"
read -p "📝 Enter your username: " USERNAME
read -p "🖥️  Enter machine name (optional, default: $(hostname)): " MACHINE_NAME
MACHINE_NAME=${MACHINE_NAME:-$(hostname)}

# ECM work parameters
read -p "🔢 Enter B1 value (default: 11000000): " B1_VALUE
B1_VALUE=${B1_VALUE:-11000000}
read -p "⭐ Enter priority filter (default: 5): " PRIORITY_VALUE
PRIORITY_VALUE=${PRIORITY_VALUE:-5}

API_ENDPOINT="https://ecm.kyleaskine.com/api/v1"
echo "🌐 Using API endpoint: $API_ENDPOINT"
echo "============================================================"
echo ""

# ============================================================
# Step 4: Setup Directory
# ============================================================
INSTALL_DIR="$HOME/ecm-wrapper"
echo "📁 Setting up in: $INSTALL_DIR"

if [ -d "$INSTALL_DIR" ]; then
    echo "⚠️  Directory exists. Removing old installation..."
    rm -rf "$INSTALL_DIR"
fi

# Clone repository
echo "📦 Cloning ecm-wrapper repository..."
git clone -q https://github.com/kyleaskine/ecm-wrapper.git "$INSTALL_DIR"
cd "$INSTALL_DIR/client"
echo "✓ Repository cloned"

# Create data directory
mkdir -p data
echo "✓ Data directory created"

# ============================================================
# Step 5: Download ECM Binary
# ============================================================
echo ""
echo "⬇️  Downloading ECM binary ($ECM_VERSION)..."
ECM_DOWNLOAD_URL="https://ecm.kyleaskine.com/downloads/${ECM_VERSION}/ecm.gz"
ECM_PATH="$HOME/ecm"

wget -q --show-progress "$ECM_DOWNLOAD_URL" -O "${ECM_PATH}.gz"
gunzip -f "${ECM_PATH}.gz"
chmod +x "$ECM_PATH"

# Verify installation
if [ -x "$ECM_PATH" ]; then
    ECM_VERSION_STR=$("$ECM_PATH" --version 2>&1 | head -1 || echo "unknown")
    echo "✓ ECM binary installed: $ECM_VERSION_STR"
else
    echo "⚠️  ECM binary download may have failed"
fi

# ============================================================
# Step 6: Install Python Dependencies
# ============================================================
echo ""
echo "📚 Installing Python dependencies..."
# Use `python3 -m pip` so packages land in the same interpreter that runs the
# client (vast.ai images often ship with conda + system Python side-by-side).
# Fall back to --break-system-packages for PEP 668 environments.
if ! python3 -m pip install requests pyyaml; then
    echo "⚠️  pip install failed, retrying with --break-system-packages..."
    python3 -m pip install --break-system-packages requests pyyaml
fi
echo "✓ Dependencies installed (requests, pyyaml)"
python3 -c "import requests, yaml; print(f'   requests {requests.__version__}, pyyaml {yaml.__version__}')"

# ============================================================
# Step 7: Detect GPU
# ============================================================
echo ""
echo "🎮 Checking for GPU..."
GPU_ENABLED="false"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_ENABLED="true"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "✓ GPU detected: $GPU_INFO"
    else
        echo "ℹ️  nvidia-smi found but no GPU detected"
    fi
else
    echo "ℹ️  No GPU detected (CPU mode)"
fi

# ============================================================
# Step 8: Create Configuration
# ============================================================
echo ""
echo "⚙️  Creating client.local.yaml..."

cat > client.local.yaml << EOF
# Cloud Instance Configuration
# Generated: $(date)

api:
  endpoint: "$API_ENDPOINT"
  timeout: 30

client:
  username: "$USERNAME"
  cpu_name: "$MACHINE_NAME"

programs:
  gmp_ecm:
    path: "$ECM_PATH"
    gpu_enabled: $GPU_ENABLED
    gpu_device: 0

# Logging configuration
logging:
  level: "INFO"
  file: "data/logs/ecm_client.log"
  console: true
EOF

echo "✓ Configuration file created"

# ============================================================
# Step 9: Setup Complete - Display Summary
# ============================================================
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE!"
echo "============================================================"
echo "Username:      $USERNAME"
echo "Machine:       $MACHINE_NAME"
echo "API Endpoint:  $API_ENDPOINT"
echo "ECM Binary:    $ECM_PATH"
echo "Architecture:  $ECM_VERSION"
echo "GPU:           $GPU_ENABLED"
echo "Working Dir:   $INSTALL_DIR/client"
echo "============================================================"
echo ""
echo "🚀 Ready to run ECM factorization!"
echo ""
echo "Example commands:"
echo ""
echo "  # Change to client directory"
echo "  cd $INSTALL_DIR/client"
echo ""
echo "  # Test with a small number (ecm_wrapper.py doesn't submit by default)"
echo "  python3 ecm_wrapper.py --composite \"123456789012345\" --curves 10 --b1 11000"
echo ""
echo "  # Auto-work mode (progressive strategy, stage 1 only)"
echo "  python3 ecm_client.py --work-type progressive --stage1-only --b1 $B1_VALUE --priority $PRIORITY_VALUE -v"
echo ""
echo "  # Auto-work with specific count"
echo "  python3 ecm_client.py --work-type progressive --work-count 10 --stage1-only --b1 $B1_VALUE --priority $PRIORITY_VALUE -v"
echo ""
if [ "$GPU_ENABLED" = "true" ]; then
echo "  # Auto-work with GPU (stage 1 only)"
echo "  python3 ecm_client.py --work-type progressive --stage1-only --b1 $B1_VALUE --priority $PRIORITY_VALUE -v"
echo ""
fi
echo "  # Auto-work with multiprocess (CPU only)"
echo "  python3 ecm_client.py --work-type progressive --multiprocess --workers 8 --stage1-only --b1 $B1_VALUE --priority $PRIORITY_VALUE -v"
echo ""
echo "============================================================"
echo ""

# Optional: Offer to start auto-work immediately
read -p "Start auto-work mode now? [y/N]: " START_NOW
if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting auto-work mode (progressive strategy)..."
    echo "Press Ctrl+C to stop"
    echo ""
    cd "$INSTALL_DIR/client"
    if [ "$GPU_ENABLED" = "true" ]; then
        python3 ecm_client.py --work-type progressive --stage1-only --b1 $B1_VALUE --priority $PRIORITY_VALUE -v
    else
        python3 ecm_client.py --work-type progressive --multiprocess --stage1-only --b1 $B1_VALUE --priority $PRIORITY_VALUE -v
    fi
fi
