"""
Token Vocabulary Management

Manages the discrete token vocabulary for POYO tokenization.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import Counter


class TokenVocabulary:
    """
    Manages vocabulary of discrete neural tokens.
    
    Features:
    - Token frequency tracking
    - Rare token filtering
    - Special tokens (PAD, MASK, etc.)
    """
    
    # Special token IDs
    PAD_TOKEN = 0
    MASK_TOKEN = 1
    UNK_TOKEN = 2
    SPECIAL_TOKENS = 3  # Number of special tokens
    
    def __init__(self, vocab_size: int = 256, min_frequency: int = 1):
        """
        Initialize vocabulary.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_frequency: Minimum token frequency to keep in vocabulary
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Token statistics
        self.token_counts: Counter = Counter()
        self.token_to_id: Dict[int, int] = {}
        self.id_to_token: Dict[int, int] = {}
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens"""
        self.token_to_id[self.PAD_TOKEN] = 0
        self.token_to_id[self.MASK_TOKEN] = 1
        self.token_to_id[self.UNK_TOKEN] = 2
        
        self.id_to_token[0] = self.PAD_TOKEN
        self.id_to_token[1] = self.MASK_TOKEN
        self.id_to_token[2] = self.UNK_TOKEN
    
    def add_tokens(self, tokens: np.ndarray) -> None:
        """
        Add tokens to vocabulary from training data.
        
        Args:
            tokens: [n_samples, n_tokens] token matrix
        """
        unique_tokens, counts = np.unique(tokens.flatten(), return_counts=True)
        for token, count in zip(unique_tokens, counts):
            self.token_counts[int(token)] += int(count)
    
    def build(self) -> None:
        """Build vocabulary from accumulated token counts"""
        # Filter by minimum frequency
        valid_tokens = [
            token for token, count in self.token_counts.items()
            if count >= self.min_frequency
        ]
        
        # Sort by frequency (most common first)
        valid_tokens = sorted(
            valid_tokens,
            key=lambda t: self.token_counts[t],
            reverse=True
        )
        
        # Take top vocab_size - SPECIAL_TOKENS
        max_tokens = self.vocab_size - self.SPECIAL_TOKENS
        valid_tokens = valid_tokens[:max_tokens]
        
        # Assign IDs
        for idx, token in enumerate(valid_tokens, start=self.SPECIAL_TOKENS):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def encode(self, token: int) -> int:
        """Convert token to vocabulary ID"""
        return self.token_to_id.get(token, self.UNK_TOKEN)
    
    def decode(self, token_id: int) -> int:
        """Convert vocabulary ID to token"""
        return self.id_to_token.get(token_id, self.UNK_TOKEN)
    
    def get_statistics(self) -> Dict:
        """Get vocabulary statistics"""
        return {
            'vocab_size': len(self.token_to_id),
            'total_tokens': sum(self.token_counts.values()),
            'most_common': self.token_counts.most_common(10),
            'coverage': len(self.token_to_id) / self.vocab_size
        }
