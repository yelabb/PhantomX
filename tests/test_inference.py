"""
Test suite for Inference and TTA components
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add phantomx to path
sys.path.insert(0, str(Path(__file__).parents[1] / 'python'))

from phantomx.vqvae import VQVAE
from phantomx.tokenizer import SpikeTokenizer
from phantomx.inference import LabramDecoder
from phantomx.tta import EntropyMinimizer, TTAOptimizer


class TestLabramDecoder:
    """Tests for LabramDecoder"""
    
    @pytest.fixture
    def decoder(self):
        """Create a test decoder"""
        model = VQVAE(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64,
            num_codes=256,
            output_dim=2
        )
        
        tokenizer = SpikeTokenizer(n_channels=142, quantization_levels=16)
        # Fit tokenizer with dummy data
        spike_trains = np.random.poisson(2.0, size=(100, 142)).astype(np.float32)
        tokenizer.fit(spike_trains)
        
        return LabramDecoder(model=model, tokenizer=tokenizer, use_tta=False)
    
    def test_decoder_initialization(self, decoder):
        """Test decoder initialization"""
        assert decoder.model is not None
        assert decoder.tokenizer is not None
        assert decoder.use_tta == False
        assert decoder.inference_count == 0
    
    def test_decode_single(self, decoder):
        """Test decoding a single sample"""
        spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
        
        kinematics = decoder.decode(spike_counts)
        
        assert kinematics.shape == (2,)
        assert decoder.inference_count == 1
    
    def test_decode_batch(self, decoder):
        """Test decoding a batch"""
        spike_batch = np.random.poisson(2.0, size=(32, 142)).astype(np.float32)
        
        kinematics = decoder.decode(spike_batch)
        
        assert kinematics.shape == (32, 2)
    
    def test_decode_with_tokens(self, decoder):
        """Test decoding with token return"""
        spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
        
        result = decoder.decode(spike_counts, return_tokens=True)
        
        assert 'kinematics' in result
        assert 'tokens' in result
        assert result['tokens'].shape == (16,)
    
    def test_decode_with_codes(self, decoder):
        """Test decoding with codebook index return"""
        spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
        
        result = decoder.decode(spike_counts, return_codes=True)
        
        assert 'kinematics' in result
        assert 'codes' in result
    
    def test_decode_batch_method(self, decoder):
        """Test large batch decoding"""
        spike_batch = np.random.poisson(2.0, size=(100, 142)).astype(np.float32)
        
        kinematics = decoder.decode_batch(spike_batch, batch_size=16)
        
        assert kinematics.shape == (100, 2)
    
    def test_get_statistics(self, decoder):
        """Test statistics retrieval"""
        # Run some inferences
        for _ in range(5):
            spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
            decoder.decode(spike_counts)
        
        stats = decoder.get_statistics()
        
        assert stats['inference_count'] == 5
        assert 'mean_latency_ms' in stats
        assert 'num_codes' in stats
    
    def test_save_load(self, decoder, tmp_path):
        """Test saving and loading decoder"""
        # Run some inference to populate state
        spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
        decoder.decode(spike_counts)
        
        # Save
        save_path = tmp_path / "decoder.pt"
        decoder.save(str(save_path))
        
        # Load
        loaded = LabramDecoder.load(str(save_path))
        
        assert loaded.model is not None
        assert loaded.tokenizer.is_fitted


class TestEntropyMinimizer:
    """Tests for EntropyMinimizer"""
    
    @pytest.fixture
    def model(self):
        return VQVAE(n_tokens=16, token_dim=256, output_dim=2)
    
    def test_initialization(self, model):
        """Test entropy minimizer initialization"""
        tta = EntropyMinimizer(model=model, learning_rate=1e-4)
        
        assert tta.model is model
        assert tta.temperature == 1.0
        assert tta.adaptation_steps == 1
    
    def test_adapt_and_decode(self, model):
        """Test TTA decoding"""
        tta = EntropyMinimizer(model=model, learning_rate=1e-4)
        
        tokens = torch.randint(0, 256, (16,))  # Single sample
        
        kinematics = tta.adapt_and_decode(tokens)
        
        assert kinematics.shape == (1, 2)
    
    def test_adapt_with_info(self, model):
        """Test TTA with info return"""
        tta = EntropyMinimizer(model=model)
        
        tokens = torch.randint(0, 256, (1, 16))
        
        kinematics, info = tta.adapt_and_decode(tokens, return_info=True)
        
        assert 'entropy_loss' in info
        assert 'kinematics_initial' in info
        assert 'kinematics_adapted' in info
    
    def test_statistics(self, model):
        """Test statistics tracking"""
        tta = EntropyMinimizer(model=model)
        
        for _ in range(3):
            tokens = torch.randint(0, 256, (1, 16))
            tta.adapt_and_decode(tokens)
        
        stats = tta.get_statistics()
        
        assert stats['n_samples_adapted'] == 3
        assert 'mean_entropy' in stats


class TestTTAOptimizer:
    """Tests for TTAOptimizer"""
    
    @pytest.fixture
    def model(self):
        return VQVAE(n_tokens=16, token_dim=256, output_dim=2)
    
    def test_entropy_strategy(self, model):
        """Test entropy minimization strategy"""
        tta = TTAOptimizer(model=model, strategy='entropy')
        
        tokens = torch.randint(0, 256, (1, 16))
        kinematics = tta.adapt_and_decode(tokens)
        
        assert kinematics.shape == (1, 2)
    
    def test_rls_strategy_no_target(self, model):
        """Test RLS strategy without target"""
        tta = TTAOptimizer(model=model, strategy='rls')
        
        tokens = torch.randint(0, 256, (1, 16))
        kinematics = tta.adapt_and_decode(tokens)
        
        assert kinematics.shape == (1, 2)
    
    def test_invalid_strategy(self, model):
        """Test that invalid strategy raises error"""
        with pytest.raises(ValueError):
            TTAOptimizer(model=model, strategy='invalid')
    
    def test_reset(self, model):
        """Test resetting TTA state"""
        tta = TTAOptimizer(model=model, strategy='entropy')
        
        # Run some adaptations
        for _ in range(3):
            tokens = torch.randint(0, 256, (1, 16))
            tta.adapt_and_decode(tokens)
        
        # Reset should not raise
        tta.reset()


class TestDecoderWithTTA:
    """Tests for LabramDecoder with TTA enabled"""
    
    def test_decoder_with_tta(self):
        """Test decoder with TTA"""
        model = VQVAE(n_tokens=16, token_dim=256, output_dim=2)
        tokenizer = SpikeTokenizer(n_channels=142)
        tokenizer.fit(np.random.poisson(2.0, size=(100, 142)).astype(np.float32))
        
        decoder = LabramDecoder(
            model=model,
            tokenizer=tokenizer,
            use_tta=True,
            tta_strategy='entropy',
            tta_lr=1e-4
        )
        
        assert decoder.use_tta
        assert decoder.tta_optimizer is not None
        
        # Decode should work
        spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
        kinematics = decoder.decode(spike_counts)
        
        assert kinematics.shape == (2,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
