import torch
import torch.nn as nn
import numpy as np
from ml.config import ModelConfig


class GeoSonarEncoder(nn.Module):
    """
    Encodes sonar scan with geometry tensor into fixed-dimension embedding.

    Input:
        - scan: (B, 1, 24, 60, 128) float32 in [0,1]
        - geo: (1, 3, 24, 60, 128) float32 — geometry buffer (broadcast over batch)

    Output:
        - (B, d_sonar) embedding

    Handles missing scans by replacing with learned embedding, properly broadcast to batch.
    """

    def __init__(self, cfg: ModelConfig, geo: np.ndarray):
        """
        Args:
            cfg: ModelConfig with d_sonar and dimension specs
            geo: Geometry tensor (24, 60, 128, 3) from build_geometry_tensor()
        """
        super().__init__()
        self.cfg = cfg

        # Register geometry as buffer (convert to torch tensor, reshape to (1, 3, 24, 60, 128))
        geo_tensor = torch.from_numpy(geo).float()  # (24, 60, 128, 3)
        geo_tensor = geo_tensor.permute(3, 0, 1, 2)  # (3, 24, 60, 128)
        geo_tensor = geo_tensor.unsqueeze(0)  # (1, 3, 24, 60, 128)
        self.register_buffer('geo', geo_tensor)

        # Step 1: Geometry projection Conv3d(3→4, kernel=1)
        self.geo_proj = nn.Conv3d(3, 4, kernel_size=1)

        # Step 3: 3D CNN stack
        # Conv3d(5→16, kernel=(3,3,7), stride=(1,1,2), pad=(1,1,3))
        self.conv1 = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=(3, 3, 7), stride=(1, 1, 2), padding=(1, 1, 3)),
            nn.BatchNorm3d(16),
            nn.GELU()
        )

        # Conv3d(16→32, kernel=(3,3,5), stride=(1,2,2), pad=(1,1,2))
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 5), stride=(1, 2, 2), padding=(1, 1, 2)),
            nn.BatchNorm3d(32),
            nn.GELU()
        )

        # Conv3d(32→64, kernel=(3,3,5), stride=(2,2,2), pad=(1,1,2))
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 5), stride=(2, 2, 2), padding=(1, 1, 2)),
            nn.BatchNorm3d(64),
            nn.GELU()
        )

        # Conv3d(64→128, kernel=(3,3,3), stride=(2,3,2), pad=(1,1,1))
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 3, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.GELU()
        )

        # Step 4: Adaptive pooling and flattening handled in forward
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Step 5: Linear projection and LayerNorm
        self.linear = nn.Linear(128, cfg.d_sonar)
        self.norm = nn.LayerNorm(cfg.d_sonar)

        # Missing scan: learned embedding for when scan is None
        self.missing_scan_embedding = nn.Parameter(torch.zeros(cfg.d_sonar))

    def forward(self, scan, batch_size=None):
        """
        Args:
            scan: (B, 1, 24, 60, 128) or None
            batch_size: Required when scan is None

        Returns:
            (B, d_sonar) embedding
        """
        # Handle missing scan case
        if scan is None:
            if batch_size is None:
                raise ValueError("batch_size required when scan is None")
            return self.missing_scan_embedding.unsqueeze(0).expand(batch_size, -1)

        B = scan.shape[0]

        # Step 1: Project geometry
        # Ensure geo is on the same device as scan
        geo = self.geo.to(scan.device) if self.geo.device != scan.device else self.geo
        geo_proj = self.geo_proj(geo)  # (1, 4, 24, 60, 128)

        # Step 2: Concatenate scan and expanded geometry projection
        geo_proj_expanded = geo_proj.expand(B, -1, -1, -1, -1)  # (B, 4, 24, 60, 128)
        x = torch.cat([scan, geo_proj_expanded], dim=1)  # (B, 5, 24, 60, 128)

        # Step 3: 3D CNN stack
        x = self.conv1(x)  # (B, 16, 24, 60, 64)
        x = self.conv2(x)  # (B, 32, 24, 30, 32)
        x = self.conv3(x)  # (B, 64, 12, 15, 16)
        x = self.conv4(x)  # (B, 128, 6, 5, 8)

        # Step 4: Adaptive pooling and flatten
        x = self.pool(x)  # (B, 128, 1, 1, 1)
        x = x.flatten(1)  # (B, 128)

        # Step 5: Linear projection and LayerNorm
        x = self.linear(x)  # (B, d_sonar)
        x = self.norm(x)  # (B, d_sonar)

        return x


class NavEncoder(nn.Module):
    """
    Encodes navigation data with cross-attention to sonar embedding.

    Input:
        - nav: (B, 7) — [east/250, north/250, depth/32, speed/15, sin(head), cos(head), confidence]
        - sonar_emb: (B, 128) from GeoSonarEncoder

    Output:
        - (B, 128) navigation-conditioned scan embedding
    """

    def __init__(self, cfg: ModelConfig):
        """
        Args:
            cfg: ModelConfig with d_nav, d_model, n_heads
        """
        super().__init__()
        self.cfg = cfg

        # Step 1: MLP projection of nav features
        # Linear(7→64) → GELU → Linear(64→d_nav) → LayerNorm(d_nav)
        self.nav_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, cfg.d_nav),
            nn.LayerNorm(cfg.d_nav)
        )

        # Step 2: Cross-attention layer
        # MultiheadAttention with embed_dim=128, num_heads=4
        self.xattn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            batch_first=True,
            dropout=cfg.dropout
        )

    def forward(self, nav, sonar_emb):
        """
        Args:
            nav: (B, 7) navigation features
            sonar_emb: (B, 128) sonar embedding from GeoSonarEncoder

        Returns:
            (B, 128) navigation-conditioned embedding
        """
        # Step 1: MLP projection of nav
        nav_emb = self.nav_mlp(nav)  # (B, 128)

        # Step 2: Cross-attention
        # Reshape for MultiheadAttention (batch_first=True)
        nav_q = nav_emb.unsqueeze(1)  # (B, 1, 128)
        sonar_kv = sonar_emb.unsqueeze(1)  # (B, 1, 128)

        # nav_q attends into sonar_kv
        out, _ = self.xattn(nav_q, sonar_kv, sonar_kv)  # (B, 1, 128)

        # Step 3: Residual connection
        out = out.squeeze(1) + nav_emb  # (B, 128)

        return out
