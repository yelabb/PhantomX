"""
Test suite for VQ-VAE components
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add phantomx to path
sys.path.insert(0, str(Path(__file__).parents[1] / 'python'))

from phantomx.vqvae import VQVAE, VQVAETrainer, Codebook, VectorQuantizer, SpikeEncoder, KinematicsDecoder


class TestCodebook:
    """Tests for Codebook and VectorQuantizer"""
    
    def test_codebook_initialization(self):
        """Test codebook initialization"""
        codebook = Codebook(num_codes=256, embedding_dim=64)
        assert codebook.num_codes == 256
        assert codebook.embedding_dim == 64
        assert codebook.embedding.weight.shape == (256, 64)
    
    def test_codebook_forward(self):
        """Test codebook forward pass"""
        codebook = Codebook(num_codes=256, embedding_dim=64)
        
        # Create fake encoder output
        z_e = torch.randn(32, 64)  # [batch, embedding_dim]
        
        z_q, indices, commitment_loss = codebook(z_e)
        
        assert z_q.shape == z_e.shape
        assert indices.shape == (32,)
        assert indices.dtype == torch.int64
        assert torch.all(indices >= 0) and torch.all(indices < 256)
        assert commitment_loss.ndim == 0  # scalar
    
    def test_vector_quantizer(self):
        """Test VectorQuantizer wrapper"""
        quantizer = VectorQuantizer(num_codes=256, embedding_dim=64)
        
        z_e = torch.randn(32, 64)
        z_q, info = quantizer(z_e)
        
        assert z_q.shape == z_e.shape
        assert 'commitment_loss' in info
        assert 'perplexity' in info
        assert 'indices' in info
        assert info['perplexity'] > 0
    
    def test_quantizer_lookup(self):
        """Test looking up codebook vectors by index"""
        quantizer = VectorQuantizer(num_codes=256, embedding_dim=64)
        
        indices = torch.randint(0, 256, (32,))
        embeddings = quantizer.lookup(indices)
        
        assert embeddings.shape == (32, 64)


class TestEncoder:
    """Tests for SpikeEncoder"""
    
    def test_encoder_initialization(self):
        """Test encoder initialization"""
        encoder = SpikeEncoder(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64
        )
        assert encoder.n_tokens == 16
        assert encoder.embedding_dim == 64
    
    def test_encoder_forward(self):
        """Test encoder forward pass"""
        encoder = SpikeEncoder(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64
        )
        
        # Input tokens
        tokens = torch.randint(0, 256, (32, 16))  # [batch, n_tokens]
        
        z_e = encoder(tokens)
        
        assert z_e.shape == (32, 64)
        assert z_e.dtype == torch.float32


class TestDecoder:
    """Tests for KinematicsDecoder"""
    
    def test_decoder_initialization(self):
        """Test decoder initialization"""
        decoder = KinematicsDecoder(
            embedding_dim=64,
            output_dim=2
        )
        assert decoder.embedding_dim == 64
        assert decoder.output_dim == 2
    
    def test_decoder_forward(self):
        """Test decoder forward pass"""
        decoder = KinematicsDecoder(
            embedding_dim=64,
            output_dim=2
        )
        
        z_q = torch.randn(32, 64)  # [batch, embedding_dim]
        
        kinematics = decoder(z_q)
        
        assert kinematics.shape == (32, 2)


class TestVQVAE:
    """Tests for full VQ-VAE model"""
    
    def test_vqvae_initialization(self):
        """Test VQ-VAE initialization"""
        model = VQVAE(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64,
            num_codes=256,
            output_dim=2
        )
        assert model.num_codes == 256
        assert model.embedding_dim == 64
    
    def test_vqvae_forward_no_targets(self):
        """Test VQ-VAE forward without targets"""
        model = VQVAE(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64,
            num_codes=256,
            output_dim=2
        )
        
        tokens = torch.randint(0, 256, (32, 16))
        
        output = model(tokens)
        
        assert 'kinematics_pred' in output
        assert 'commitment_loss' in output
        assert 'perplexity' in output
        assert 'indices' in output
        assert output['kinematics_pred'].shape == (32, 2)
    
    def test_vqvae_forward_with_targets(self):
        """Test VQ-VAE forward with targets"""
        model = VQVAE(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64,
            num_codes=256,
            output_dim=2
        )
        
        tokens = torch.randint(0, 256, (32, 16))
        targets = torch.randn(32, 2)
        
        output = model(tokens, targets)
        
        assert 'reconstruction_loss' in output
        assert 'total_loss' in output
        assert output['total_loss'].requires_grad
    
    def test_vqvae_encode(self):
        """Test VQ-VAE encoding"""
        model = VQVAE(n_tokens=16, token_dim=256)
        tokens = torch.randint(0, 256, (32, 16))
        
        indices = model.encode(tokens)
        
        assert indices.shape == (32,)
        assert torch.all(indices >= 0) and torch.all(indices < model.num_codes)
    
    def test_vqvae_decode(self):
        """Test VQ-VAE decoding from indices"""
        model = VQVAE(n_tokens=16, token_dim=256)
        
        indices = torch.randint(0, model.num_codes, (32,))
        
        kinematics = model.decode(indices)
        
        assert kinematics.shape == (32, 2)
    
    def test_vqvae_training_step(self):
        """Test that gradients flow correctly"""
        model = VQVAE(n_tokens=16, token_dim=256)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        tokens = torch.randint(0, 256, (32, 16))
        targets = torch.randn(32, 2)
        
        # Forward
        output = model(tokens, targets)
        loss = output['total_loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        assert loss.item() > 0


class TestVQVAETrainer:
    """Tests for VQ-VAE Trainer"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        model = VQVAE(n_tokens=16, token_dim=256)
        trainer = VQVAETrainer(model=model, learning_rate=3e-4)
        
        assert trainer.model is model
        assert trainer.device is not None
    
    def test_trainer_evaluate(self):
        """Test trainer evaluation"""
        from torch.utils.data import DataLoader, TensorDataset
        
        model = VQVAE(n_tokens=16, token_dim=256)
        trainer = VQVAETrainer(model=model)
        
        # Create dummy dataset
        tokens = torch.randint(0, 256, (100, 16))
        kinematics = torch.randn(100, 2)
        
        dataset = TensorDataset(tokens, kinematics)
        
        # Wrap in dict-returning format
        class DictDataset:
            def __init__(self, tokens, kinematics):
                self.tokens = tokens
                self.kinematics = kinematics
            
            def __len__(self):
                return len(self.tokens)
            
            def __getitem__(self, idx):
                return {
                    'tokens': self.tokens[idx],
                    'kinematics': self.kinematics[idx]
                }
        
        dataset = DictDataset(tokens, kinematics)
        loader = DataLoader(dataset, batch_size=32)
        
        metrics = trainer.evaluate(loader)
        
        assert 'loss' in metrics
        assert 'reconstruction_loss' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
