from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

EXPECTED_PROOF_IMAGES = {
    "segmentation_diagonal_2d.png",
    "segmentation_gaussian_2d.png",
    "splitter_tests/phase_1_axis_threshold.png",
    "splitter_tests/phase_2_diagonal_2d.png",
    "splitter_tests/phase_3_gaussian_2d.png",
    "splitter_tests/phase_4_periodic_wraparound.png",
    "splitter_tests/phase_5_fuzzy_boundary.png",
    "splitter_tests/phase_6_linear_leaf.png",
    "splitter_tests/phase_7_sparse_set.png",
    "splitter_tests/phase_8_learning_rate_shrinkage.png",
}


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
    actual = {
        path.relative_to(asset_dir).as_posix()
        for path in asset_dir.glob("**/*.png")
        if path.relative_to(asset_dir).as_posix() in EXPECTED_PROOF_IMAGES
    }
    assert actual == EXPECTED_PROOF_IMAGES
    for relative_path in sorted(EXPECTED_PROOF_IMAGES):
        _assert_nonblank_segmentation(asset_dir / relative_path)
