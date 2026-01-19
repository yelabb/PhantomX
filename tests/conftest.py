"""
PyTest configuration for PhantomX tests
"""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch


# Add phantomx to Python path
project_root = Path(__file__).parent.parent
python_path = project_root / "python"
sys.path.insert(0, str(python_path))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Get compute device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_spikes():
    """Generate sample spike counts"""
    return np.random.poisson(2.0, size=(100, 142)).astype(np.float32)


@pytest.fixture
def sample_tokens():
    """Generate sample tokens"""
    return torch.randint(0, 256, (32, 16))


@pytest.fixture
def sample_kinematics():
    """Generate sample kinematics"""
    return torch.randn(32, 2)
