#!/bin/bash
set -e

PROJECT_DIR="project"
VENV_DIR="$PROJECT_DIR-venv"
DATA_DIR="data"

echo "=== PhantomX GPU Post-Init ==="
cd ~

# Create venv if needed
if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Activating venv..."
source $VENV_DIR/bin/activate

# Install packages if not done
if ! pip show torch &> /dev/null; then
    echo "Installing PyTorch with CUDA..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install numpy scipy scikit-learn h5py pynwb jupyter
fi

# Copy project files
mkdir -p $PROJECT_DIR
cp -r /app/python/* $PROJECT_DIR/
cp /app/requirements.txt $PROJECT_DIR/

# Create data directory
mkdir -p $DATA_DIR

echo ""
echo "=== GPU Setup Complete ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""
echo "To run experiments, SSH in with: fly ssh console"
echo "Then: cd ~/project && source ~/project-venv/bin/activate"
echo ""

# Keep running for SSH access
sleep inf
