from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


def _assert_nonblank_segmentation(path: Path) -> None:
    assert path.exists()
    image = mpimg.imread(path)
    assert image.ndim in {2, 3}
    assert image.shape[0] >= 400
    assert image.shape[1] >= 400

    rgb = image[..., :3] if image.ndim == 3 else image
    assert float(np.std(rgb)) > 0.01

    flattened = rgb.reshape(-1, rgb.shape[-1]) if rgb.ndim == 3 else rgb.reshape(-1, 1)
    sample = flattened[:: max(1, len(flattened) // 2048)]
    unique_colors = np.unique(np.round(sample, 3), axis=0)
    assert len(unique_colors) >= 16


def test_segmentation_proof_images_are_generated_and_nonblank():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_segmentation_proofs.py"

    subprocess.run([sys.executable, str(script)], check=True, cwd=repo_root)

    asset_dir = repo_root / "docs" / "assets"
    _assert_nonblank_segmentation(asset_dir / "segmentation_diagonal_2d.png")
    _assert_nonblank_segmentation(asset_dir / "segmentation_gaussian_2d.png")
