from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Geometry / scan
    n_az: int = 24
    n_beam: int = 60
    n_range: int = 128
    scan_max_range_m: float = 40.0
    beam_min_deg: float = 5.0
    beam_max_deg: float = 64.0
    az_half_deg: float = 30.0

    # Model dimensions (REDUCED - 10x smaller for better generalization)
    d_sonar: int = 64        # SonarEncoder output dim (was 128)
    d_nav: int = 64          # NavEncoder output dim (was 128)
    d_model: int = 64        # Transformer hidden dim (was 128)
    n_heads: int = 2         # Reduced from 4
    d_ff: int = 256          # Reduced from 512
    n_layers: int = 2        # Reduced from 4
    dropout: float = 0.3     # Increased from 0.1 for regularization

    # Sequence
    window_local: int = 10   # short-range attention window (ticks)
    window_long: int = 180   # long-range context (ticks, stride 6 → 30 tokens)
    window_size: int = 60    # total dataset window for training

    # Species
    species: tuple = ("largemouth bass", "rainbow trout", "common carp", "bluegill bream")
    n_species: int = 4

    # Training
    horizon_s: float = 300.0
    batch_size: int = 32
    lr_peak: float = 1e-3        # Increased from 3e-4 for faster learning
    lr_finetune: float = 5e-5
    weight_decay: float = 1e-2   # Increased from 1e-4 for stronger regularization
    grad_clip: float = 0.5       # Tighter clipping to stabilize training
    epochs_pretrain: int = 30
    epochs_finetune: int = 20
    val_fraction: float = 0.15
    pos_weight_cap: float = 50.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
