"""
FSQ-VAE: Finite Scalar Quantization VAE with Dual-Head Decoder

Architecture:
    Spikes → Encoder → z_e → FSQ → z_q → [Decoder A: Kinematics]
                                        [Decoder B: Spike Reconstruction]

Key Innovations:
1. FSQ replaces VQ with topology-preserving scalar quantization
2. Dual-head decoder prevents "supervised bottleneck" trap
3. No commitment loss, codebook collapse, or EMA updates required

Loss:
    L = L_velocity + λ * L_reconstruction
    
    Where:
    - L_velocity: MSE loss for kinematics prediction
    - L_reconstruction: Poisson NLL for spike reconstruction
    - λ: Balancing weight (default 0.5)

Theoretical Justification (Information Bottleneck):
    We want to compress X (spikes) into Z such that:
    - I(Z; Y) is maximized (velocity prediction)
    - I(Z; X) is maintained (spike reconstruction)
    
    This creates a robust, generalizable neural representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .encoder import SpikeEncoder, TransformerEncoder
from .fsq import FSQ, ResidualFSQ
from .decoder import KinematicsDecoder, RecurrentDecoder
from .spike_decoder import SpikeReconstructionDecoder, TokenReconstructionDecoder


class FSQVAE(nn.Module):
    """
    FSQ-VAE with Dual-Head Decoder
    
    A foundation model architecture that:
    1. Uses FSQ for topology-preserving discrete representations
    2. Predicts kinematics (velocity) for the downstream task
    3. Reconstructs spikes to maintain rich neural representations
    """
    
    def __init__(
        self,
        # Input settings
        n_tokens: int = 16,
        token_dim: int = 256,
        
        # FSQ settings
        fsq_levels: List[int] = [8, 5, 5, 5],
        embedding_dim: int = 64,
        use_residual_fsq: bool = False,
        n_residual_fsq: int = 3,
        
        # Encoder settings
        encoder_hidden_dims: List[int] = None,
        use_transformer_encoder: bool = False,
        
        # Kinematics decoder settings (Head A)
        decoder_hidden_dims: List[int] = None,
        output_dim: int = 2,  # (vx, vy)
        use_recurrent_decoder: bool = False,
        
        # Spike reconstruction settings (Head B)
        n_channels: int = 96,
        n_bins: int = 10,
        spike_decoder_hidden_dims: List[int] = None,
        reconstruction_type: str = 'spike',  # 'spike' or 'token'
        
        # Loss settings
        reconstruction_weight: float = 0.5,  # λ in the loss function
        
        dropout: float = 0.1
    ):
        """
        Initialize FSQ-VAE.
        
        Args:
            n_tokens: Number of input tokens
            token_dim: Token vocabulary size
            fsq_levels: Number of levels per FSQ dimension
            embedding_dim: Latent embedding dimension
            use_residual_fsq: Whether to use Residual FSQ
            n_residual_fsq: Number of residual FSQ layers
            encoder_hidden_dims: Encoder layer dimensions
            use_transformer_encoder: Use transformer instead of MLP
            decoder_hidden_dims: Kinematics decoder layer dimensions
            output_dim: Output dimension (2 for vx, vy)
            use_recurrent_decoder: Use LSTM decoder for kinematics
            n_channels: Number of neural channels for spike reconstruction
            n_bins: Number of time bins for spike reconstruction
            spike_decoder_hidden_dims: Spike decoder layer dimensions
            reconstruction_type: 'spike' for rate reconstruction, 'token' for token prediction
            reconstruction_weight: Weight for reconstruction loss (λ)
            dropout: Dropout rate
        """
        super().__init__()
        
        # Default hidden dimensions
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256, 256]
        if spike_decoder_hidden_dims is None:
            spike_decoder_hidden_dims = [128, 256, 256]
        
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_type = reconstruction_type
        self.output_dim = output_dim
        self.use_recurrent_decoder = use_recurrent_decoder
        
        # ======== ENCODER ========
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
        
        # ======== FSQ QUANTIZER ========
        if use_residual_fsq:
            self.quantizer = ResidualFSQ(
                levels=fsq_levels,
                input_dim=embedding_dim,
                n_residual=n_residual_fsq
            )
        else:
            self.quantizer = FSQ(
                levels=fsq_levels,
                input_dim=embedding_dim
            )
        
        # ======== DECODER A: KINEMATICS (Task Head) ========
        if use_recurrent_decoder:
            self.kinematics_decoder = RecurrentDecoder(
                embedding_dim=embedding_dim,
                hidden_dim=decoder_hidden_dims[0],
                output_dim=output_dim,
                dropout=dropout
            )
        else:
            self.kinematics_decoder = KinematicsDecoder(
                embedding_dim=embedding_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=output_dim,
                dropout=dropout
            )
        
        # ======== DECODER B: SPIKE RECONSTRUCTION (Foundation Head) ========
        if reconstruction_type == 'spike':
            self.reconstruction_decoder = SpikeReconstructionDecoder(
                embedding_dim=embedding_dim,
                hidden_dims=spike_decoder_hidden_dims,
                n_channels=n_channels,
                n_bins=n_bins,
                dropout=dropout
            )
        elif reconstruction_type == 'token':
            self.reconstruction_decoder = TokenReconstructionDecoder(
                embedding_dim=embedding_dim,
                hidden_dims=spike_decoder_hidden_dims,
                n_tokens=n_tokens,
                vocab_size=token_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown reconstruction_type: {reconstruction_type}")
        
        # Store configuration
        self.config = {
            'n_tokens': n_tokens,
            'token_dim': token_dim,
            'fsq_levels': fsq_levels,
            'embedding_dim': embedding_dim,
            'output_dim': output_dim,
            'n_channels': n_channels,
            'n_bins': n_bins,
            'reconstruction_weight': reconstruction_weight,
            'reconstruction_type': reconstruction_type,
        }
    
    def forward(
        self,
        tokens: torch.Tensor,
        kinematics_targets: Optional[torch.Tensor] = None,
        reconstruction_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FSQ-VAE.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            kinematics_targets: [batch, output_dim] target kinematics
            reconstruction_targets: Target for reconstruction:
                - If spike: [batch, n_channels, n_bins] spike counts
                - If token: [batch, n_tokens] token IDs
            
        Returns:
            Dictionary containing predictions, losses, and statistics
        """
        # ======== ENCODE ========
        z_e = self.encoder(tokens)  # [batch, embedding_dim]
        
        # ======== QUANTIZE (FSQ) ========
        z_q, quant_info = self.quantizer(z_e)
        
        # ======== DECODE: KINEMATICS (Head A) ========
        if self.use_recurrent_decoder:
            z_q_seq = z_q.unsqueeze(1)
            kinematics_pred, _ = self.kinematics_decoder(z_q_seq)
            kinematics_pred = kinematics_pred.squeeze(1)
        else:
            kinematics_pred = self.kinematics_decoder(z_q)
        
        # ======== DECODE: RECONSTRUCTION (Head B) ========
        if self.reconstruction_type == 'spike':
            reconstruction_pred = self.reconstruction_decoder(z_q, return_rates=True)
        else:
            reconstruction_pred = self.reconstruction_decoder(z_q)
        
        # ======== PREPARE OUTPUT ========
        output = {
            'kinematics_pred': kinematics_pred,
            'reconstruction_pred': reconstruction_pred,
            'indices': quant_info['indices'],
            'perplexity': quant_info['perplexity'],
            'codebook_usage': quant_info['codebook_usage'],
            'z_e': z_e,
            'z_q': z_q,
        }
        
        # Include FSQ-specific info
        if 'z_fsq' in quant_info:
            output['z_fsq'] = quant_info['z_fsq']
        
        # ======== COMPUTE LOSSES ========
        if kinematics_targets is not None:
            kinematics_loss = F.mse_loss(kinematics_pred, kinematics_targets)
            output['kinematics_loss'] = kinematics_loss
        
        if reconstruction_targets is not None:
            if self.reconstruction_type == 'spike':
                reconstruction_loss = self.reconstruction_decoder.compute_loss(
                    z_q, reconstruction_targets
                )
            else:
                reconstruction_loss = self.reconstruction_decoder.compute_loss(
                    z_q, reconstruction_targets
                )
            output['reconstruction_loss'] = reconstruction_loss
        
        # Compute total loss if both targets provided
        if kinematics_targets is not None and reconstruction_targets is not None:
            total_loss = (
                kinematics_loss 
                + self.reconstruction_weight * reconstruction_loss
            )
            output['total_loss'] = total_loss
        elif kinematics_targets is not None:
            # Fallback: only kinematics loss
            output['total_loss'] = kinematics_loss
        
        return output
    
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens to discrete code indices.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            
        Returns:
            indices: Discrete code indices
        """
        z_e = self.encoder(tokens)
        _, quant_info = self.quantizer(z_e)
        return quant_info['indices']
    
    def decode_kinematics(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete code indices to kinematics.
        
        Args:
            indices: Discrete code indices
            
        Returns:
            kinematics: [batch, output_dim] predicted kinematics
        """
        z_q = self.quantizer.lookup(indices)
        
        if self.use_recurrent_decoder:
            z_q_seq = z_q.unsqueeze(1)
            kinematics, _ = self.kinematics_decoder(z_q_seq)
            kinematics = kinematics.squeeze(1)
        else:
            kinematics = self.kinematics_decoder(z_q)
        
        return kinematics
    
    def get_latent_representation(
        self,
        tokens: torch.Tensor,
        quantized: bool = True
    ) -> torch.Tensor:
        """
        Get latent representation for analysis.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            quantized: If True, return quantized z_q; else return z_e
            
        Returns:
            z: [batch, embedding_dim] latent representation
        """
        z_e = self.encoder(tokens)
        
        if quantized:
            z_q, _ = self.quantizer(z_e)
            return z_q
        else:
            return z_e
    
    @property
    def codebook_size(self) -> int:
        """Get effective codebook size"""
        return self.quantizer.codebook_size
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.quantizer.embedding_dim


class FSQVAEWithCausalTransformer(FSQVAE):
    """
    FSQ-VAE variant that uses a causal transformer encoder.
    
    This is the architecture recommended in the critique:
    "Encoder: Keep your Causal Transformer (it's working)"
    """
    
    def __init__(
        self,
        # Transformer settings
        n_channels: int = 96,
        n_bins: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        
        # FSQ settings
        fsq_levels: List[int] = [8, 5, 5, 5],
        embedding_dim: int = 64,
        
        # Decoder settings
        decoder_hidden_dims: List[int] = None,
        output_dim: int = 2,
        
        # Reconstruction settings
        spike_decoder_hidden_dims: List[int] = None,
        reconstruction_weight: float = 0.5,
        
        dropout: float = 0.1
    ):
        """
        Initialize FSQ-VAE with Causal Transformer.
        
        This version takes raw spike data instead of tokens.
        """
        # Don't call parent __init__, we build our own architecture
        nn.Module.__init__(self)
        
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [128, 256, 256]
        if spike_decoder_hidden_dims is None:
            spike_decoder_hidden_dims = [128, 256, 256]
        
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.reconstruction_weight = reconstruction_weight
        self.output_dim = output_dim
        self.reconstruction_type = 'spike'
        self.use_recurrent_decoder = False
        
        # ======== CAUSAL TRANSFORMER ENCODER ========
        self.input_projection = nn.Linear(n_channels, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Generate causal mask for transformer
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(n_bins, n_bins), diagonal=1).bool()
        )
        
        # Aggregate transformer output to single vector
        self.aggregator = nn.Sequential(
            nn.Linear(d_model * n_bins, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # ======== FSQ QUANTIZER ========
        self.quantizer = FSQ(
            levels=fsq_levels,
            input_dim=embedding_dim
        )
        
        # ======== DECODER A: KINEMATICS ========
        self.kinematics_decoder = KinematicsDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # ======== DECODER B: SPIKE RECONSTRUCTION ========
        self.reconstruction_decoder = SpikeReconstructionDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=spike_decoder_hidden_dims,
            n_channels=n_channels,
            n_bins=n_bins,
            dropout=dropout
        )
        
        self.config = {
            'n_channels': n_channels,
            'n_bins': n_bins,
            'd_model': d_model,
            'fsq_levels': fsq_levels,
            'embedding_dim': embedding_dim,
            'output_dim': output_dim,
            'reconstruction_weight': reconstruction_weight,
        }
    
    def encode_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Encode raw spike data with causal transformer.
        
        Args:
            spikes: [batch, n_channels, n_bins] spike counts
            
        Returns:
            z_e: [batch, embedding_dim] continuous latent
        """
        # Reshape: [batch, n_bins, n_channels]
        x = spikes.permute(0, 2, 1)
        
        # Project to d_model
        x = self.input_projection(x)  # [batch, n_bins, d_model]
        
        # Causal transformer
        x = self.transformer(x, mask=self.causal_mask)  # [batch, n_bins, d_model]
        
        # Flatten and aggregate
        x = x.reshape(x.shape[0], -1)  # [batch, n_bins * d_model]
        z_e = self.aggregator(x)  # [batch, embedding_dim]
        
        return z_e
    
    def forward(
        self,
        spikes: torch.Tensor,
        kinematics_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with raw spike input.
        
        Args:
            spikes: [batch, n_channels, n_bins] spike counts
            kinematics_targets: [batch, output_dim] target kinematics
            
        Returns:
            Dictionary with predictions and losses
        """
        # ======== ENCODE ========
        z_e = self.encode_spikes(spikes)
        
        # ======== QUANTIZE ========
        z_q, quant_info = self.quantizer(z_e)
        
        # ======== DECODE: KINEMATICS ========
        kinematics_pred = self.kinematics_decoder(z_q)
        
        # ======== DECODE: RECONSTRUCTION ========
        reconstruction_pred = self.reconstruction_decoder(z_q, return_rates=True)
        
        # ======== OUTPUT ========
        output = {
            'kinematics_pred': kinematics_pred,
            'reconstruction_pred': reconstruction_pred,
            'indices': quant_info['indices'],
            'perplexity': quant_info['perplexity'],
            'codebook_usage': quant_info['codebook_usage'],
            'z_e': z_e,
            'z_q': z_q,
        }
        
        if 'z_fsq' in quant_info:
            output['z_fsq'] = quant_info['z_fsq']
        
        # ======== LOSSES ========
        if kinematics_targets is not None:
            kinematics_loss = F.mse_loss(kinematics_pred, kinematics_targets)
            output['kinematics_loss'] = kinematics_loss
            
            # Reconstruction loss (spikes as targets)
            reconstruction_loss = self.reconstruction_decoder.compute_loss(
                z_q, spikes
            )
            output['reconstruction_loss'] = reconstruction_loss
            
            # Total loss
            total_loss = (
                kinematics_loss 
                + self.reconstruction_weight * reconstruction_loss
            )
            output['total_loss'] = total_loss
        
        return output
