#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Setting up Multi-Agent Debate Engine Environment..."

# 1. Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "✅ Virtual environment 'venv' already exists."
fi

# 3. Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip and install dependencies
echo "📥 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎉 Setup complete!"
echo "To activate the environment in your shell, run:"
echo "    source venv/bin/activate"
