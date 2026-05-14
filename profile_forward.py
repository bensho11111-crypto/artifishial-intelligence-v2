"""Profile where time is actually spent in model.forward().

Confirms whether the 60-tick Python loop is the bottleneck, not memory leaks.
"""
import sys
import time

import torch

sys.path.insert(0, "src")

from ml.config import ModelConfig
from ml.model import FishCatchTransformer


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = ModelConfig()
    model = FishCatchTransformer(cfg)
    model.eval()  # no dropout, deterministic timing

    B, T = 32, cfg.window_size  # match real training batch
    scans = torch.rand(B, T, 1, cfg.n_az, cfg.n_beam, cfg.n_range)
    scan_valid = torch.ones(B, T, dtype=torch.bool)
    nav = torch.rand(B, T, 7)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model(scans, scan_valid, nav)

    # Time full forward
    print(f"\nFull forward pass (B={B}, T={T}):")
    n_runs = 3
    for i in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            _ = model(scans, scan_valid, nav)
        print(f"  run {i+1}: {time.time()-t0:.2f}s")

    # Isolate the 60-tick sonar loop
    print(f"\nIsolated: 60 sequential sonar_encoder calls (one sample-batch per tick):")
    for i in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            for t in range(T):
                _ = model.sonar_encoder(scans[:, t])
        print(f"  run {i+1}: {time.time()-t0:.2f}s")

    # Vectorized: single big sonar_encoder call
    print(f"\nVectorized: 1 sonar_encoder call on (B*T, 1, ...) = ({B*T}, 1, ...):")
    flat_scans = scans.reshape(B * T, 1, cfg.n_az, cfg.n_beam, cfg.n_range)
    for i in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            _ = model.sonar_encoder(flat_scans)
        print(f"  run {i+1}: {time.time()-t0:.2f}s")

    # Backward pass (where the "stall" really lives)
    print(f"\nFull forward + backward + step (B={B}, T={T}):")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for i in range(n_runs):
        opt.zero_grad()
        t0 = time.time()
        logits = model(scans, scan_valid, nav)
        loss = logits.sum()
        loss.backward()
        opt.step()
        print(f"  run {i+1}: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
