#!/bin/bash

echo "Setting up AdvisoryAI Jarvis..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
EOF
    echo "⚠️  Please edit .env and add your Azure OpenAI credentials"
    echo ""
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Generate mock data if it doesn't exist
if [ ! -f mock_data.json ]; then
    echo ""
    echo "Generating mock data..."
    python3 data_generator.py
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your Azure OpenAI credentials:"
echo "   - AZURE_OPENAI_ENDPOINT"
echo "   - AZURE_OPENAI_API_KEY"
echo "2. Run the app: streamlit run app.py"
echo "   Or use CLI: python main.py"

