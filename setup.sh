#!/bin/bash
# Setup script for MCPulse

set -e

echo "🔌 MCPulse Setup Script"
echo "======================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please edit .env file and add your API keys:"
    echo "   - OPENAI_API_KEY or ANTHROPIC_API_KEY"
    echo "   - MongoDB settings (if using MongoDB)"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create config directory if it doesn't exist
mkdir -p config
echo "✓ Config directory ready"
echo ""

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Edit .env file with your API keys"
echo "  3. Run: python main.py"
echo ""
echo "The application will be available at http://localhost:7860"
