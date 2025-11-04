#!/bin/bash
set -e

SCRIPT_NAME="aicode"
INSTALL_PATH="$HOME/.local/bin/$SCRIPT_NAME"
PROJECT_DIR="/Volumes/home/aicode"
LIMA_INSTANCE="aicode"

# Self-install function
self_install() {
    if [ "$0" != "$INSTALL_PATH" ]; then
        echo "📦 Installing $SCRIPT_NAME to $INSTALL_PATH..."
        mkdir -p "$HOME/.local/bin"
        cp "$0" "$INSTALL_PATH"
        chmod +x "$INSTALL_PATH"
        echo "✅ Installed! You can now run: $SCRIPT_NAME"
        echo "💡 Make sure $HOME/.local/bin is in your PATH"
        
        # Check if in PATH
        if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
            echo "⚠️  Add this to your ~/.zshrc or ~/.bashrc:"
            echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
        exit 0
    fi
}

# Check if --install flag is passed
if [ "$1" == "--install" ]; then
    self_install
fi

# Navigate to project directory
cd "$PROJECT_DIR" || { echo "❌ Directory not found: $PROJECT_DIR"; exit 1; }

# Check if Lima instance exists and is running
if limactl list | grep -q "^$LIMA_INSTANCE.*Running"; then
    echo "✅ Lima instance '$LIMA_INSTANCE' is already running"
    exec limactl shell "$LIMA_INSTANCE" bash -c "cd '$PROJECT_DIR' && exec bash"
elif limactl list | grep -q "^$LIMA_INSTANCE"; then
    echo "🔄 Lima instance '$LIMA_INSTANCE' exists but not running, starting..."
    limactl start "$LIMA_INSTANCE"
    exec limactl shell "$LIMA_INSTANCE" bash -c "cd '$PROJECT_DIR' && exec bash"
else
    echo "🚀 Creating new Lima instance '$LIMA_INSTANCE'..."
    
    # Create a minimal Lima template for Apple Silicon
    cat > /tmp/aicode.yaml <<EOF
# Minimal aicode configuration for Apple Silicon
images:
  - location: "https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-arm64.img"
    arch: "aarch64"

# ONLY mount the project directory - no defaults
mounts:
  - location: "$PROJECT_DIR"
    writable: true

mountType: "reverse-sshfs"

cpus: 4
memory: "4GiB"
disk: "10GiB"

# Ensure mount point exists
provision:
  - mode: system
    script: |
      mkdir -p "$PROJECT_DIR"
EOF

    # Start with the minimal template
    limactl start --name="$LIMA_INSTANCE" /tmp/aicode.yaml
    
    echo "✅ Lima instance created successfully"
fi

# Open shell in project directory
exec limactl shell "$LIMA_INSTANCE" bash -c "cd '$PROJECT_DIR' && exec bash"
