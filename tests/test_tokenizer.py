"""
Test suite for POYO Spike Tokenizer
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add phantomx to path
sys.path.insert(0, str(Path(__file__).parents[1] / 'python'))

from phantomx.tokenizer import SpikeTokenizer, TokenVocabulary, PopulationBinner


class TestSpikeTokenizer:
    """Tests for SpikeTokenizer"""
    
    def test_initialization(self):
        """Test tokenizer initialization with default parameters"""
        tokenizer = SpikeTokenizer()
        assert tokenizer.config.n_channels == 142
        assert tokenizer.config.quantization_levels == 16
        assert not tokenizer.is_fitted
    
    def test_initialization_custom(self):
        """Test tokenizer with custom parameters"""
        tokenizer = SpikeTokenizer(
            n_channels=64,
            quantization_levels=8,
            use_population_norm=False
        )
        assert tokenizer.config.n_channels == 64
        assert tokenizer.config.quantization_levels == 8
        assert tokenizer.config.use_population_norm == False
    
    def test_fit(self):
        """Test fitting tokenizer on training data"""
        tokenizer = SpikeTokenizer(n_channels=142)
        
        # Generate fake spike data
        spike_trains = np.random.poisson(2.0, size=(1000, 142)).astype(np.float32)
        
        # Fit should work and update state
        tokenizer.fit(spike_trains)
        
        assert tokenizer.is_fitted
        assert tokenizer.mean is not None
        assert tokenizer.std is not None
        assert tokenizer.mean.shape == (142,)
        assert tokenizer.std.shape == (142,)
    
    def test_tokenize_single(self):
        """Test tokenizing a single sample"""
        tokenizer = SpikeTokenizer(n_channels=142, quantization_levels=16)
        
        # Fit first
        spike_trains = np.random.poisson(2.0, size=(100, 142)).astype(np.float32)
        tokenizer.fit(spike_trains)
        
        # Tokenize single sample
        spike_counts = np.random.poisson(2.0, size=142).astype(np.float32)
        tokens = tokenizer.tokenize(spike_counts)
        
        assert tokens.shape == (16,)
        assert tokens.dtype == np.int32
        assert np.all(tokens >= 0)
        assert np.all(tokens < 256)
    
    def test_tokenize_batch(self):
        """Test tokenizing a batch of samples"""
        tokenizer = SpikeTokenizer(n_channels=142, quantization_levels=16)
        
        # Fit first
        spike_trains = np.random.poisson(2.0, size=(100, 142)).astype(np.float32)
        tokenizer.fit(spike_trains)
        
        # Tokenize batch
        batch_size = 32
        spike_batch = np.random.poisson(2.0, size=(batch_size, 142)).astype(np.float32)
        tokens = tokenizer.tokenize(spike_batch)
        
        assert tokens.shape == (batch_size, 16)
        assert tokens.dtype == np.int32
    
    def test_dropout_invariance(self):
        """Test that tokenizer handles electrode dropout gracefully"""
        tokenizer = SpikeTokenizer(
            n_channels=142,
            quantization_levels=16,
            dropout_invariant=True
        )
        
        spike_trains = np.random.poisson(2.0, size=(100, 142)).astype(np.float32)
        tokenizer.fit(spike_trains)
        
        # Full spike counts
        full_spikes = np.random.poisson(2.0, size=142).astype(np.float32)
        
        # Simulate 50% electrode dropout
        dropped_spikes = full_spikes.copy()
        dropout_mask = np.random.random(142) < 0.5
        dropped_spikes[dropout_mask] = 0.0
        
        # Both should tokenize without error
        tokens_full = tokenizer.tokenize(full_spikes)
        tokens_dropped = tokenizer.tokenize(dropped_spikes)
        
        assert tokens_full.shape == tokens_dropped.shape
    
    def test_save_load(self, tmp_path):
        """Test saving and loading tokenizer state"""
        tokenizer = SpikeTokenizer(n_channels=142)
        spike_trains = np.random.poisson(2.0, size=(100, 142)).astype(np.float32)
        tokenizer.fit(spike_trains)
        
        # Save
        save_path = tmp_path / "tokenizer.pt"
        tokenizer.save(str(save_path))
        
        # Load
        loaded = SpikeTokenizer.load(str(save_path))
        
        assert loaded.is_fitted
        assert np.allclose(loaded.mean, tokenizer.mean)
        assert np.allclose(loaded.std, tokenizer.std)


class TestTokenVocabulary:
    """Tests for TokenVocabulary"""
    
    def test_initialization(self):
        """Test vocabulary initialization"""
        vocab = TokenVocabulary(vocab_size=256)
        assert vocab.vocab_size == 256
        assert TokenVocabulary.PAD_TOKEN == 0
        assert TokenVocabulary.MASK_TOKEN == 1
        assert TokenVocabulary.UNK_TOKEN == 2
    
    def test_add_tokens(self):
        """Test adding tokens to vocabulary"""
        vocab = TokenVocabulary(vocab_size=256)
        
        tokens = np.random.randint(0, 100, size=(100, 16))
        vocab.add_tokens(tokens)
        
        assert len(vocab.token_counts) > 0
    
    def test_build_and_encode(self):
        """Test building vocabulary and encoding tokens"""
        vocab = TokenVocabulary(vocab_size=256)
        
        tokens = np.random.randint(0, 100, size=(1000, 16))
        vocab.add_tokens(tokens)
        vocab.build()
        
        # Encode known token
        encoded = vocab.encode(50)
        assert encoded >= 0
        assert encoded < vocab.vocab_size
    
    def test_encode_unknown_token(self):
        """Test encoding unknown token returns UNK"""
        vocab = TokenVocabulary(vocab_size=256)
        tokens = np.array([[1, 2, 3, 4]])
        vocab.add_tokens(tokens)
        vocab.build()
        
        # Token that was never seen
        encoded = vocab.encode(9999)
        assert encoded == TokenVocabulary.UNK_TOKEN


class TestPopulationBinner:
    """Tests for PopulationBinner"""
    
    def test_initialization(self):
        """Test binner initialization"""
        binner = PopulationBinner(bin_size_ms=25.0, sampling_rate_hz=40.0)
        assert binner.bin_size_ms == 25.0
        assert binner.sampling_rate_hz == 40.0
    
    def test_bin_spikes(self):
        """Test spike binning"""
        binner = PopulationBinner(bin_size_ms=25.0, sampling_rate_hz=40.0)
        
        # Generate random spike times
        n_spikes = 1000
        duration_s = 10.0
        spike_times = np.random.uniform(0, duration_s, size=n_spikes)
        spike_channels = np.random.randint(0, 142, size=n_spikes)
        
        binned = binner.bin_spikes(spike_times, spike_channels, n_channels=142, duration_s=duration_s)
        
        n_expected_bins = int(duration_s * 1000.0 / 25.0)
        assert binned.shape == (n_expected_bins, 142)
    
    def test_compute_population_rate(self):
        """Test population rate computation"""
        binner = PopulationBinner()
        
        spike_counts = np.random.poisson(2.0, size=(100, 142))
        pop_rate = binner.compute_population_rate(spike_counts)
        
        assert pop_rate.shape == (100,)
        assert np.all(pop_rate >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
