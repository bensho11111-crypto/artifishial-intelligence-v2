"""
src/ml/model.py

FishCatchTransformer: Multi-scale temporal transformer for fish catch prediction.

Combines:
1. Per-tick embeddings from GeoSonarEncoder + NavEncoder
2. Two-scale temporal attention (local + long-range)
3. Species cross-attention head
"""
import torch
import torch.nn as nn
import numpy as np
from ml.config import ModelConfig
from ml.geometry import build_geometry_tensor
from ml.encoders import GeoSonarEncoder, NavEncoder


def causal_mask(size: int) -> torch.Tensor:
    """
    Returns (size, size) boolean mask; True positions are masked (not attended to).

    Creates an upper triangular matrix where True indicates positions that should
    be masked out in causal attention (no attention to future tokens).

    Args:
        size: Sequence length

    Returns:
        (size, size) boolean tensor with True above diagonal
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


class FishCatchTransformer(nn.Module):
    """
    Two-scale temporal transformer for fish catch prediction.

    Architecture:
    1. Per-tick embeddings from GeoSonarEncoder + NavEncoder with positional encoding
    2. Local stream: last 10 ticks with causal attention
    3. Long-range stream: stride-6 sampling of last 180 ticks with causal attention
    4. Species cross-attention head: (n_species, 128) queries attending to combined output

    Forward pass:
        scans:      (B, T, 1, 24, 60, 128) or None per tick
        scan_valid: (B, T) boolean mask for valid scans
        nav:        (B, T, 7) navigation features
        -> (B, 4) logits for each species
    """

    def __init__(self, cfg: ModelConfig):
        """
        Initialize FishCatchTransformer.

        Args:
            cfg: ModelConfig with all hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        # Register geometry tensor as non-trainable buffer
        geo = build_geometry_tensor(cfg)
        geo_tensor = torch.from_numpy(geo).float()  # (24, 60, 128, 3)
        geo_tensor = geo_tensor.permute(3, 0, 1, 2)  # (3, 24, 60, 128)
        self.register_buffer('_geo', geo_tensor)

        # Encoders
        self.sonar_encoder = GeoSonarEncoder(cfg, geo)
        self.nav_encoder = NavEncoder(cfg)

        # Per-tick embedding layers
        self.input_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.window_size + 1, cfg.d_model)  # +1 for safety

        # Two-scale temporal attention streams
        encoder_layer_local = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer_local, num_layers=cfg.n_layers)

        encoder_layer_lr = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.lr_transformer = nn.TransformerEncoder(encoder_layer_lr, num_layers=cfg.n_layers)

        # Combine streams
        self.combine_proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.combine_norm = nn.LayerNorm(cfg.d_model)

        # Species cross-attention head
        self.species_queries = nn.Embedding(cfg.n_species, cfg.d_model)
        self.xattn_head = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            batch_first=True,
            dropout=cfg.dropout
        )

        # Output projection to logits
        self.logits_proj = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        scans: torch.Tensor,
        scan_valid: torch.BoolTensor,
        nav: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for FishCatchTransformer.

        Args:
            scans:      (B, T, 1, 24, 60, 128) sonar scans or None
            scan_valid: (B, T) boolean mask indicating valid scans
            nav:        (B, T, 7) navigation features

        Returns:
            (B, 4) logits for species classification
        """
        B, T = nav.shape[0], nav.shape[1]

        # Step 1: Per-tick embeddings
        # Process sonar scans frame-by-frame
        sonar_embs = []
        for t in range(T):
            if scan_valid[:, t].all():
                # All scans valid at this tick
                s_emb = self.sonar_encoder(scans[:, t])  # (B, d_sonar)
            else:
                # Some scans missing; use learned embedding
                s_emb = self.sonar_encoder(None, batch_size=B)  # (B, d_sonar)
            sonar_embs.append(s_emb)

        sonar_embs = torch.stack(sonar_embs, dim=0)  # (T, B, d_sonar)

        # Fuse sonar and nav frame-by-frame
        per_tick_fused = []
        for t in range(T):
            fused = self.nav_encoder(nav[:, t], sonar_embs[t])  # (B, d_model)
            per_tick_fused.append(fused)

        tick_emb = torch.stack(per_tick_fused, dim=0)  # (T, B, d_model)

        # Add positional embeddings
        pos_indices = torch.arange(T, device=tick_emb.device)
        pos_emb_vals = self.pos_emb(pos_indices)  # (T, d_model)
        tick_emb = tick_emb + pos_emb_vals.unsqueeze(1)  # (T, B, d_model)

        # Project and normalize per-tick embeddings
        tick_emb = self.input_proj(tick_emb)  # (T, B, d_model)

        # Convert to (B, T, d_model) for transformer
        tick_emb = tick_emb.permute(1, 0, 2)  # (B, T, d_model)

        # Step 2: Two-scale temporal attention streams
        # Local stream: last 10 ticks
        local_in = tick_emb[:, -self.cfg.window_local:, :]  # (B, min(T, 10), d_model)
        local_size = min(T, self.cfg.window_local)
        local_mask = causal_mask(local_size).to(tick_emb.device)
        local_out = self.local_transformer(local_in, mask=local_mask)  # (B, local_size, d_model)
        local_out = local_out[:, -1, :]  # (B, d_model) — take last token

        # Long-range stream: stride-6 sampling of last 180 ticks
        # Target: 30 tokens (180 / 6), but pad if T < 180
        lr_stride = 6
        lr_window = 180
        lr_target_size = lr_window // lr_stride  # 30

        # Extract stride-6 tokens from entire sequence (not just last 180)
        # This handles variable T gracefully
        if T >= lr_window:
            # Take last lr_window ticks, then stride
            lr_in_full = tick_emb[:, -lr_window:, :]  # (B, lr_window, d_model)
            lr_in = lr_in_full[:, ::lr_stride, :]  # (B, 30, d_model)
        else:
            # Stride from beginning
            lr_in = tick_emb[:, ::lr_stride, :]  # (B, ceil(T/6), d_model)

            # Pad to target size if needed
            current_size = lr_in.shape[1]
            if current_size < lr_target_size:
                pad_size = lr_target_size - current_size
                padding = torch.zeros(B, pad_size, self.cfg.d_model, device=tick_emb.device)
                lr_in = torch.cat([lr_in, padding], dim=1)  # (B, 30, d_model)

        lr_size = lr_in.shape[1]
        lr_mask = causal_mask(lr_size).to(tick_emb.device)
        lr_out = self.lr_transformer(lr_in, mask=lr_mask)  # (B, lr_size, d_model)
        lr_out = lr_out[:, -1, :]  # (B, d_model) — take last token

        # Step 3: Combine streams
        combined = torch.cat([local_out, lr_out], dim=-1)  # (B, 2*d_model)
        combined = self.combine_proj(combined)  # (B, d_model)
        combined = self.combine_norm(combined)  # (B, d_model)

        # Step 4: Species cross-attention head
        # Species queries attend to combined representation
        species_q = self.species_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, n_species, d_model)
        species_kv = combined.unsqueeze(1)  # (B, 1, d_model)

        species_out, _ = self.xattn_head(species_q, species_kv, species_kv)  # (B, n_species, d_model)

        # Project to logits
        logits = self.logits_proj(species_out).squeeze(-1)  # (B, n_species)

        return logits
