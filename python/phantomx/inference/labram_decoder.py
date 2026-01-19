"""
LaBraM Runtime Decoder

High-level interface for zero-shot inference and test-time adaptation.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Union
import numpy as np

from ..vqvae import VQVAE
from ..tokenizer import SpikeTokenizer
from ..tta import TTAOptimizer


class LabramDecoder:
    """
    High-level decoder interface for runtime inference.
    
    Features:
    - Zero-shot decoding from pre-trained codebook
    - Optional test-time adaptation
    - Electrode dropout handling
    - Efficient batched inference
    
    Example:
        decoder = LabramDecoder.load("models/mc_maze_codebook.pt")
        
        for packet in live_stream:
            spikes = packet.spikes.spike_counts  # [142]
            velocity = decoder.decode(spikes)     # {vx, vy}
    """
    
    def __init__(
        self,
        model: VQVAE,
        tokenizer: SpikeTokenizer,
        use_tta: bool = False,
        tta_strategy: str = 'entropy',
        tta_lr: float = 1e-4,
        device: str = None
    ):
        """
        Initialize decoder.
        
        Args:
            model: Pre-trained VQ-VAE model
            tokenizer: Fitted spike tokenizer
            use_tta: Enable test-time adaptation
            tta_strategy: TTA strategy ('entropy', 'rls', 'hybrid')
            tta_lr: Learning rate for TTA
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_tta = use_tta
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # TTA optimizer
        if use_tta:
            self.tta_optimizer = TTAOptimizer(
                model=model,
                strategy=tta_strategy,
                learning_rate=tta_lr,
                device=device
            )
        else:
            self.tta_optimizer = None
        
        # Statistics
        self.inference_count = 0
        self.latency_history = []
    
    def decode(
        self,
        spike_counts: Union[np.ndarray, torch.Tensor],
        return_tokens: bool = False,
        return_codes: bool = False
    ) -> Union[np.ndarray, Dict]:
        """
        Decode spike counts to kinematics.
        
        Args:
            spike_counts: [n_channels] or [batch, n_channels] spike counts
            return_tokens: If True, also return discrete tokens
            return_codes: If True, also return codebook indices
            
        Returns:
            kinematics: [2] or [batch, 2] velocity (vx, vy)
            OR dictionary with kinematics, tokens, codes (if flags set)
        """
        import time
        start_time = time.time()
        
        # Convert to numpy if needed
        if isinstance(spike_counts, torch.Tensor):
            spike_counts = spike_counts.cpu().numpy()
        
        # Handle batching
        is_batch = spike_counts.ndim == 2
        if not is_batch:
            spike_counts = spike_counts[np.newaxis, :]
        
        # Tokenize
        tokens = self.tokenizer.tokenize(spike_counts)
        tokens_torch = torch.from_numpy(tokens).long().to(self.device)
        
        # Decode
        if self.use_tta and self.tta_optimizer is not None:
            # TTA needs gradients, don't use no_grad
            kinematics = self.tta_optimizer.adapt_and_decode(tokens_torch)
            codes = None
        else:
            with torch.no_grad():
                output = self.model(tokens_torch)
                kinematics = output['kinematics_pred']
                codes = output['indices'] if return_codes else None
        
        # Convert to numpy
        kinematics_np = kinematics.detach().cpu().numpy()
        
        if not is_batch:
            kinematics_np = kinematics_np[0]
        
        # Track latency
        latency = (time.time() - start_time) * 1000  # ms
        self.latency_history.append(latency)
        self.inference_count += 1
        
        # Return
        if return_tokens or return_codes:
            result = {'kinematics': kinematics_np}
            if return_tokens:
                result['tokens'] = tokens if is_batch else tokens[0]
            if return_codes:
                codes_np = codes.cpu().numpy()
                result['codes'] = codes_np if is_batch else codes_np[0]
            return result
        
        return kinematics_np
    
    def decode_batch(
        self,
        spike_batch: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Decode a large batch of spike counts efficiently.
        
        Args:
            spike_batch: [n_samples, n_channels] spike counts
            batch_size: Batch size for processing
            
        Returns:
            kinematics: [n_samples, 2] predicted velocities
        """
        n_samples = len(spike_batch)
        kinematics_list = []
        
        for i in range(0, n_samples, batch_size):
            batch = spike_batch[i:i+batch_size]
            kinematics = self.decode(batch)
            kinematics_list.append(kinematics)
        
        return np.concatenate(kinematics_list, axis=0)
    
    def get_codebook_embedding(self, code_idx: int) -> np.ndarray:
        """
        Get codebook embedding vector by index.
        
        Args:
            code_idx: Codebook index
            
        Returns:
            embedding: [embedding_dim] codebook vector
        """
        embeddings = self.model.get_codebook_embeddings()
        return embeddings[code_idx].cpu().numpy()
    
    def get_statistics(self) -> Dict:
        """Get decoder statistics"""
        stats = {
            'inference_count': self.inference_count,
            'mean_latency_ms': np.mean(self.latency_history) if self.latency_history else 0.0,
            'std_latency_ms': np.std(self.latency_history) if self.latency_history else 0.0,
            'num_codes': self.model.num_codes,
            'embedding_dim': self.model.embedding_dim,
            'use_tta': self.use_tta
        }
        
        if self.use_tta and self.tta_optimizer is not None:
            stats.update(self.tta_optimizer.get_statistics())
        
        return stats
    
    def reset_tta(self):
        """Reset test-time adaptation state"""
        if self.tta_optimizer is not None:
            self.tta_optimizer.reset()
    
    def save(self, path: str) -> None:
        """
        Save decoder checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_state': {
                'config': self.tokenizer.config.__dict__,
                'mean': self.tokenizer.mean,
                'std': self.tokenizer.std,
                'is_fitted': self.tokenizer.is_fitted
            },
            'use_tta': self.use_tta
        }
        torch.save(checkpoint, path)
        print(f"Decoder saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = None, use_tta: bool = False) -> "LabramDecoder":
        """
        Load pre-trained decoder.
        
        Args:
            path: Path to checkpoint
            device: Device to load on
            use_tta: Enable test-time adaptation
            
        Returns:
            Loaded decoder instance
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Reconstruct model (architecture must match)
        # TODO: Save model config in checkpoint for full reconstruction
        model = VQVAE(
            n_tokens=16,
            token_dim=256,
            embedding_dim=64,
            num_codes=256,
            output_dim=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Reconstruct tokenizer
        tokenizer = SpikeTokenizer(**checkpoint['tokenizer_state']['config'])
        tokenizer.mean = checkpoint['tokenizer_state']['mean']
        tokenizer.std = checkpoint['tokenizer_state']['std']
        tokenizer.is_fitted = checkpoint['tokenizer_state']['is_fitted']
        
        # Create decoder
        decoder = cls(
            model=model,
            tokenizer=tokenizer,
            use_tta=use_tta,
            device=device
        )
        
        print(f"Decoder loaded from {path}")
        print(f"  Codebook size: {model.num_codes}")
        print(f"  Embedding dim: {model.embedding_dim}")
        print(f"  TTA enabled: {use_tta}")
        
        return decoder
