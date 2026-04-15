#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================"
echo "    Setting up Vectera Project"
echo "======================================"

echo ""
echo "Checking prerequisites..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' is not installed."
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Or visit: https://docs.astral.sh/uv/"
    exit 1
fi
echo "✅ 'uv' is installed."

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: 'npm' is not installed."
    echo "Please install Node.js and npm."
    echo "Or visit: https://nodejs.org/"
    exit 1
fi
echo "✅ 'npm' is installed."

echo ""
echo "Installing Python dependencies..."
# Use uv sync to install the dependencies in the virtual environment
uv sync

echo ""
echo "Installing Node dependencies (LiteParse)..."
# Install liteparse globally as required by the pipeline
if npm install -g @llamaindex/liteparse; then
    echo "✅ @llamaindex/liteparse installed successfully."
else
    echo "⚠️  Warning: Failed to install @llamaindex/liteparse globally."
    echo "This might require elevated privileges. Try running:"
    echo "sudo npm install -g @llamaindex/liteparse"
    exit 1
fi

echo ""
echo "======================================"
echo "    Setup completed successfully!"
echo "======================================"
echo "You can now start the application with:"
echo "    uv run streamlit run streamlit_app.py"
