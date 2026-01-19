"""
VQ-VAE Main Architecture

Combines encoder, vector quantizer, and decoder.
"""

import torch
import torch.nn as nn
from typing import Dict

from .encoder import SpikeEncoder, TransformerEncoder
from .codebook import VectorQuantizer
from .decoder import KinematicsDecoder, RecurrentDecoder


class VQVAE(nn.Module):
    """
    LaBraM Vector-Quantized Variational Autoencoder
    
    Architecture:
        Tokens → Encoder → z_e → Quantizer → z_q → Decoder → Kinematics
                                      ↓
                                  Codebook (K codes)
    
    Loss:
        L = ||kinematics - target||² + β ||sg[z_e] - e||² + ||z_e - sg[e]||²
        
        Where:
        - First term: Reconstruction loss
        - Second term: Codebook loss
        - Third term: Commitment loss
        - sg[·]: Stop gradient
        - e: Nearest codebook embedding
    """
    
    def __init__(
        self,
        # Tokenizer settings
        n_tokens: int = 16,
        token_dim: int = 256,
        
        # Latent space settings
        embedding_dim: int = 64,
        num_codes: int = 256,
        commitment_cost: float = 0.25,
        
        # Encoder/decoder settings
        encoder_hidden_dims: list = None,
        decoder_hidden_dims: list = None,
        output_dim: int = 2,
        
        # Architecture choices
        use_transformer_encoder: bool = False,
        use_recurrent_decoder: bool = False,
        
        dropout: float = 0.1
    ):
        """
        Initialize VQ-VAE.
        
        Args:
            n_tokens: Number of input tokens
            token_dim: Vocabulary size
            embedding_dim: Latent code dimension
            num_codes: Number of codebook entries
            commitment_cost: Weight for commitment loss (β)
            encoder_hidden_dims: Encoder layer dimensions
            decoder_hidden_dims: Decoder layer dimensions
            output_dim: Output dimension (2 for vx, vy)
            use_transformer_encoder: Use transformer instead of MLP encoder
            use_recurrent_decoder: Use LSTM/GRU decoder
            dropout: Dropout rate
        """
        super().__init__()
        
        # Default hidden dimensions
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256, 256]
        
        # Encoder
        if use_transformer_encoder:
            self.encoder = TransformerEncoder(
                token_dim=token_dim,
                embedding_dim=embedding_dim,
                dropout=dropout
            )
        else:
            self.encoder = SpikeEncoder(
                n_tokens=n_tokens,
                token_dim=token_dim,
                hidden_dims=encoder_hidden_dims,
                embedding_dim=embedding_dim,
                dropout=dropout
            )
        
        # Vector quantizer with codebook
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        # Decoder
        if use_recurrent_decoder:
            self.decoder = RecurrentDecoder(
                embedding_dim=embedding_dim,
                hidden_dim=decoder_hidden_dims[0],
                output_dim=output_dim,
                dropout=dropout
            )
        else:
            self.decoder = KinematicsDecoder(
                embedding_dim=embedding_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=output_dim,
                dropout=dropout
            )
        
        self.use_recurrent_decoder = use_recurrent_decoder
        self.output_dim = output_dim
    
    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VQ-VAE.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            targets: [batch, output_dim] target kinematics (for computing loss)
            
        Returns:
            Dictionary containing:
                - kinematics_pred: [batch, output_dim] predicted kinematics
                - commitment_loss: Scalar commitment loss
                - perplexity: Codebook utilization metric
                - indices: [batch] discrete code indices
                - reconstruction_loss: MSE loss (if targets provided)
                - total_loss: Combined loss (if targets provided)
        """
        # Encode tokens to continuous latents
        z_e = self.encoder(tokens)  # [batch, embedding_dim]
        
        # Vector quantization
        z_q, quant_info = self.quantizer(z_e)
        
        # Decode to kinematics
        if self.use_recurrent_decoder:
            # Add sequence dimension for recurrent decoder
            z_q_seq = z_q.unsqueeze(1)  # [batch, 1, embedding_dim]
            kinematics_pred, _ = self.decoder(z_q_seq)
            kinematics_pred = kinematics_pred.squeeze(1)  # [batch, output_dim]
        else:
            kinematics_pred = self.decoder(z_q)  # [batch, output_dim]
        
        # Prepare output
        output = {
            'kinematics_pred': kinematics_pred,
            'commitment_loss': quant_info['commitment_loss'],
            'perplexity': quant_info['perplexity'],
            'indices': quant_info['indices'],
            'z_e': z_e,
            'z_q': z_q
        }
        
        # Compute reconstruction loss if targets provided
        if targets is not None:
            reconstruction_loss = nn.functional.mse_loss(kinematics_pred, targets)
            total_loss = reconstruction_loss + quant_info['commitment_loss']
            
            output['reconstruction_loss'] = reconstruction_loss
            output['total_loss'] = total_loss
        
        return output
    
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens to discrete code indices.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            
        Returns:
            indices: [batch] discrete code indices
        """
        z_e = self.encoder(tokens)
        _, quant_info = self.quantizer(z_e)
        return quant_info['indices']
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete code indices to kinematics.
        
        Args:
            indices: [batch] discrete code indices
            
        Returns:
            kinematics: [batch, output_dim] predicted kinematics
        """
        # Look up codebook vectors
        z_q = self.quantizer.lookup(indices)
        
        # Decode to kinematics
        if self.use_recurrent_decoder:
            z_q_seq = z_q.unsqueeze(1)
            kinematics, _ = self.decoder(z_q_seq)
            kinematics = kinematics.squeeze(1)
        else:
            kinematics = self.decoder(z_q)
        
        return kinematics
    
    def get_codebook_embeddings(self) -> torch.Tensor:
        """
        Get all codebook embeddings.
        
        Returns:
            embeddings: [num_codes, embedding_dim] codebook vectors
        """
        return self.quantizer.codebook.embedding.weight.data
    
    @property
    def num_codes(self) -> int:
        """Number of codebook entries"""
        return self.quantizer.codebook.num_codes
    
    @property
    def embedding_dim(self) -> int:
        """Latent embedding dimension"""
        return self.quantizer.codebook.embedding_dim
