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

    # Model dimensions
    d_sonar: int = 128       # SonarEncoder output dim
    d_nav: int = 128         # NavEncoder output dim
    d_model: int = 128       # Transformer hidden dim
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    dropout: float = 0.1

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
    lr_peak: float = 3e-4
    lr_finetune: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs_pretrain: int = 30
    epochs_finetune: int = 20
    val_fraction: float = 0.15
    pos_weight_cap: float = 50.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
