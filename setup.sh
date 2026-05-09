#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================"
echo "    Setting up RAG Project"
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

echo ""
echo "Installing Python dependencies..."
# Use uv sync to install the dependencies in the virtual environment
uv sync

echo ""
echo "======================================"
echo "    Setup completed successfully!"
echo "======================================"
echo "You can now start the application with:"
echo "    uv run streamlit run streamlit_app.py"
