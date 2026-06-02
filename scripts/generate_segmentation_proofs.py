#!/usr/bin/env python3
"""Generate deterministic PNG proof images for GeoBoost spatial segmentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from geoboost import GeoBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"


def train_diagonal() -> GeoBoostRegressor:
    points = []
    target = []
    for x in np.linspace(-3.0, 3.0, 25):
        for y in np.linspace(-3.0, 3.0, 25):
            points.append([float(x), float(y)])
            target.append(10.0 if x + y > 0.0 else -10.0)
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=2,
        splitters=["diagonal_2d"],
        backend="rust",
    )
    model.fit(points, target)
    return model


def train_radial() -> GeoBoostRegressor:
    points = []
    target = []
    for x in np.linspace(-3.0, 3.0, 31):
        for y in np.linspace(-3.0, 3.0, 31):
            points.append([float(x), float(y)])
            target.append(10.0 if np.hypot(x, y) <= 1.5 else -10.0)
    model = GeoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=2,
        splitters=["gaussian_2d"],
        backend="rust",
    )
    model.fit(points, target)
    return model


def plot_segmentation(model: GeoBoostRegressor, path: Path, title: str) -> None:
    grid = np.linspace(-3.0, 3.0, 180)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    pred = model.predict(points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    mesh = ax.contourf(xx, yy, pred, levels=24, cmap="coolwarm", alpha=0.92)
    ax.contour(xx, yy, pred, colors="black", linewidths=0.7, levels=3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax, label="prediction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_segmentation(
        train_diagonal(),
        OUT / "segmentation_diagonal_2d.png",
        "GeoBoost diagonal_2d learned segmentation",
    )
    plot_segmentation(
        train_radial(),
        OUT / "segmentation_gaussian_2d.png",
        "GeoBoost gaussian_2d learned segmentation",
    )


if __name__ == "__main__":
    main()
