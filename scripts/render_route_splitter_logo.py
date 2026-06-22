from __future__ import annotations

# ruff: noqa: B905,E501,I001

import csv
import io
import math
import re
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "tmp/avatar-history/warcraft-avatar-history.zip"
OUT = ROOT / "static/img/cartoboost-route-splitter-logo.svg"

DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/mylesoneill/warcraft-avatar-history"

SIZE = 512
PANEL_W = 176
PANEL_H = 402
LEFT_X = 54
RIGHT_X = 282
PANEL_Y = 44
PLOT_PAD_X = 8
PLOT_PAD_Y = 14

RAW_BOUNDS = {
    "Kalimdor": (0.0, 1.0, 0.0, 1.0),
    "Eastern Kingdoms": (0.0, 1.0, 0.0, 1.0),
}

TREE_COLORS = [
    "#27c2b5",
    "#65c9ff",
    "#ffb454",
    "#ff5f7a",
    "#9bdf8a",
    "#b891ff",
    "#f57ac2",
    "#f6d365",
]

MAX_RENDERED_ZONES = 64
MAX_RENDERED_EDGES = 130
AVATAR_GRAPH_CACHE: tuple[list[Node], list[tuple[str, str, str, float]]] | None = None


@dataclass
class Node:
    key: str
    name: str
    faction: str
    continent: str
    zone: str
    x: float
    y: float
    routes: list[str]
    timers: dict[str, float]
    risk: float = 0.0
    leaf: int = 0


@dataclass
class Leaf:
    x0: float
    y0: float
    x1: float
    y1: float
    value: float
    index: int


@dataclass
class Split:
    axis: str
    threshold: float
    x0: float
    y0: float
    x1: float
    y1: float
    depth: int


@dataclass
class ObliqueSplit:
    a: float
    b: float
    threshold: float
    score: float


@dataclass
class GaussianSplit:
    cx: float
    cy: float
    radius: float
    score: float


def ensure_inputs() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def clean_zone(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def csv_rows(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        for row in reader:
            yield {key.strip(): value.strip() for key, value in row.items() if key is not None}


def load_zone_continents(archive: zipfile.ZipFile) -> dict[str, str]:
    continents: dict[str, str] = {}
    for row in csv_rows(archive, "zones.csv"):
        continent = row.get("Continent", "")
        if continent in {"Kalimdor", "Eastern Kingdoms"}:
            continents[clean_zone(row["Zone_Name"])] = continent
    return continents


def zone_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower().replace("the ", ""))


def load_zone_coordinates(
    archive: zipfile.ZipFile, zone_continents: dict[str, str]
) -> dict[str, tuple[float, float]]:
    by_continent = {"0": "Eastern Kingdoms", "1": "Kalimdor"}
    candidates: dict[str, list[tuple[float, float]]] = defaultdict(list)
    keyed_zones = [(zone, zone_key(zone), continent) for zone, continent in zone_continents.items()]
    for row in csv_rows(archive, "location_coords.csv"):
        continent = by_continent.get(row.get("Map_ID", ""))
        if continent is None:
            continue
        location_name = clean_zone(row.get("Location_Name", ""))
        location_key = zone_key(location_name.split(":", 1)[0])
        full_key = zone_key(location_name)
        try:
            x_coord = float(row["X_coord"])
            y_coord = float(row["Y_coord"])
        except (KeyError, ValueError):
            continue
        for zone, key, zone_continent in keyed_zones:
            if zone_continent != continent:
                continue
            if key and (key == location_key or key in full_key):
                candidates[zone].append((x_coord, y_coord))
                break
    coords: dict[str, tuple[float, float]] = {}
    for zone, values in candidates.items():
        values = sorted(values)
        mid = len(values) // 2
        xs = sorted(x for x, _ in values)
        ys = sorted(y for _, y in values)
        coords[zone] = (xs[mid], ys[mid])
    return coords


def normalize_coordinates(raw_nodes: list[tuple[str, str, float, float, int, float]]) -> list[Node]:
    bounds: dict[str, tuple[float, float, float, float]] = {}
    for continent in {"Kalimdor", "Eastern Kingdoms"}:
        continent_nodes = [row for row in raw_nodes if row[1] == continent]
        xs = [row[2] for row in continent_nodes]
        ys = [row[3] for row in continent_nodes]
        bounds[continent] = (min(xs), max(xs), min(ys), max(ys))
    nodes: list[Node] = []
    for zone, continent, x_coord, y_coord, count, avg_level in raw_nodes:
        min_x, max_x, min_y, max_y = bounds[continent]
        nx = (x_coord - min_x) / (max_x - min_x) if max_x > min_x else 0.5
        ny = 1.0 - ((y_coord - min_y) / (max_y - min_y) if max_y > min_y else 0.5)
        node = Node(
            key=f"{continent}:{zone}",
            name=zone,
            faction="AvatarHistory",
            continent=continent,
            zone=zone,
            x=min(1.0, max(0.0, nx)),
            y=min(1.0, max(0.0, ny)),
            routes=[],
            timers={},
        )
        # Store observation count and average level in fields the legacy renderer
        # does not otherwise use, so the visual remains data-driven.
        node.timers = {"observations": float(count), "avg_level": avg_level}
        nodes.append(node)
    return nodes


def load_avatar_history_graph() -> tuple[list[Node], list[tuple[str, str, str, float]]]:
    global AVATAR_GRAPH_CACHE
    if AVATAR_GRAPH_CACHE is not None:
        return AVATAR_GRAPH_CACHE
    with zipfile.ZipFile(DATA_PATH) as archive:
        zone_continents = load_zone_continents(archive)
        zone_coords = load_zone_coordinates(archive, zone_continents)
        valid_zones = set(zone_continents) & set(zone_coords)
        counts: dict[str, int] = defaultdict(int)
        level_sums: dict[str, float] = defaultdict(float)
        transitions: dict[tuple[str, str], int] = defaultdict(int)
        last_zone_by_char: dict[str, str] = {}
        for row in csv_rows(archive, "wowah_data.csv"):
            zone = clean_zone(row.get("zone", ""))
            if zone not in valid_zones:
                continue
            char = row.get("char", "")
            counts[zone] += 1
            try:
                level_sums[zone] += float(row.get("level", "0"))
            except ValueError:
                pass
            previous = last_zone_by_char.get(char)
            if previous and previous != zone and previous in valid_zones:
                a, b = sorted((previous, zone))
                transitions[(a, b)] += 1
            last_zone_by_char[char] = zone

        ranked_zones = [
            zone
            for zone, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)
            if zone in zone_coords
        ][:MAX_RENDERED_ZONES]
        rendered = set(ranked_zones)
        raw_nodes = [
            (
                zone,
                zone_continents[zone],
                zone_coords[zone][0],
                zone_coords[zone][1],
                counts[zone],
                level_sums[zone] / counts[zone],
            )
            for zone in ranked_zones
        ]
        nodes = normalize_coordinates(raw_nodes)
        by_zone = {node.name: node for node in nodes}
        edges = [
            (
                by_zone[a].key,
                by_zone[b].key,
                "movement",
                float(weight),
            )
            for (a, b), weight in sorted(
                transitions.items(), key=lambda item: item[1], reverse=True
            )
            if a in rendered and b in rendered
        ][:MAX_RENDERED_EDGES]
        AVATAR_GRAPH_CACHE = (nodes, edges)
        return nodes, edges


def parse_nodes() -> list[Node]:
    nodes, _ = load_avatar_history_graph()
    return nodes


def display_point(node: Node) -> tuple[float, float]:
    panel_x = LEFT_X if node.continent == "Kalimdor" else RIGHT_X
    min_x, max_x, min_y, max_y = RAW_BOUNDS[node.continent]
    nx = (node.x - min_x) / (max_x - min_x)
    ny = (node.y - min_y) / (max_y - min_y)
    x = panel_x + PLOT_PAD_X + nx * (PANEL_W - PLOT_PAD_X * 2)
    y = PANEL_Y + PLOT_PAD_Y + ny * (PANEL_H - PLOT_PAD_Y * 2)
    return x, y


def graph_edges(nodes: list[Node]) -> tuple[list[tuple[str, str, str, float]], dict[str, Node]]:
    by_key = {node.key: node for node in nodes}
    _, edges = load_avatar_history_graph()
    return edges, by_key


def compute_risk(
    nodes: list[Node], edges: list[tuple[str, str, str, float]], by_key: dict[str, Node]
) -> None:
    degree = {node.key: 0 for node in nodes}
    movement = {node.key: 0.0 for node in nodes}
    for a, b, _, _ in edges:
        degree[a] += 1
        degree[b] += 1
    for a, b, _, weight in edges:
        movement[a] += weight
        movement[b] += weight
    max_degree = max(degree.values()) or 1
    max_observations = max(node.timers["observations"] for node in nodes)
    max_movement = max(movement.values()) or 1.0
    for node in nodes:
        population = math.log1p(node.timers["observations"]) / math.log1p(max_observations)
        level = min(1.0, node.timers["avg_level"] / 70.0)
        hub = degree[node.key] / max_degree
        flow = math.log1p(movement[node.key]) / math.log1p(max_movement)
        node.risk = min(1.0, 0.08 + 0.42 * population + 0.24 * flow + 0.20 * hub + 0.14 * level)


def sse(items: list[Node]) -> float:
    if not items:
        return 0.0
    mean = sum(node.risk for node in items) / len(items)
    return sum((node.risk - mean) ** 2 for node in items)


def fit_tree(
    items: list[Node],
    bounds: tuple[float, float, float, float],
    depth: int,
    max_depth: int,
    leaves: list[Leaf],
    splits: list[Split],
) -> None:
    x0, y0, x1, y1 = bounds
    if depth == max_depth or len(items) < 5:
        leaves.append(
            Leaf(x0, y0, x1, y1, sum(node.risk for node in items) / max(1, len(items)), len(leaves))
        )
        return
    parent = sse(items)
    best = None
    for axis in ["x", "y"]:
        idx = 0 if axis == "x" else 1
        values = sorted({display_point(node)[idx] for node in items})
        for a, b in zip(values, values[1:]):
            threshold = (a + b) / 2
            left = [node for node in items if display_point(node)[idx] <= threshold]
            right = [node for node in items if display_point(node)[idx] > threshold]
            if len(left) < 3 or len(right) < 3:
                continue
            gain = parent - sse(left) - sse(right)
            if best is None or gain > best[0]:
                best = (gain, axis, threshold, left, right)
    if best is None or best[0] < 0.01:
        leaves.append(
            Leaf(x0, y0, x1, y1, sum(node.risk for node in items) / max(1, len(items)), len(leaves))
        )
        return
    _, axis, threshold, left, right = best
    splits.append(Split(axis, threshold, x0, y0, x1, y1, depth))
    if axis == "x":
        fit_tree(left, (x0, y0, threshold, y1), depth + 1, max_depth, leaves, splits)
        fit_tree(right, (threshold, y0, x1, y1), depth + 1, max_depth, leaves, splits)
    else:
        fit_tree(left, (x0, y0, x1, threshold), depth + 1, max_depth, leaves, splits)
        fit_tree(right, (x0, threshold, x1, y1), depth + 1, max_depth, leaves, splits)


def fit_oblique_split(nodes: list[Node]) -> ObliqueSplit:
    parent = sse(nodes)
    best: ObliqueSplit | None = None
    directions = [
        (1.0, 1.0),
        (1.0, -1.0),
        (0.72, 1.0),
        (1.0, -0.62),
        (0.45, 1.0),
        (1.0, 0.35),
    ]
    for raw_a, raw_b in directions:
        norm = math.hypot(raw_a, raw_b)
        a = raw_a / norm
        b = raw_b / norm
        projected = sorted(
            {a * display_point(node)[0] + b * display_point(node)[1] for node in nodes}
        )
        for low, high in zip(projected, projected[1:]):
            threshold = (low + high) / 2
            left = [
                node
                for node in nodes
                if a * display_point(node)[0] + b * display_point(node)[1] <= threshold
            ]
            right = [
                node
                for node in nodes
                if a * display_point(node)[0] + b * display_point(node)[1] > threshold
            ]
            if len(left) < 8 or len(right) < 8:
                continue
            gain = parent - sse(left) - sse(right)
            if best is None or gain > best.score:
                best = ObliqueSplit(a, b, threshold, gain)
    if best is None:
        return ObliqueSplit(1 / math.sqrt(2), -1 / math.sqrt(2), 256, 0)
    return best


def fit_gaussian_split(nodes: list[Node]) -> GaussianSplit:
    parent = sse(nodes)
    best: GaussianSplit | None = None
    centers = sorted(nodes, key=lambda node: node.risk, reverse=True)[:16]
    for center in centers:
        cx, cy = display_point(center)
        distances = sorted(
            {math.hypot(display_point(node)[0] - cx, display_point(node)[1] - cy) for node in nodes}
        )
        for low, high in zip(distances, distances[1:]):
            radius = (low + high) / 2
            inside = [
                node
                for node in nodes
                if math.hypot(display_point(node)[0] - cx, display_point(node)[1] - cy) <= radius
            ]
            outside = [
                node
                for node in nodes
                if math.hypot(display_point(node)[0] - cx, display_point(node)[1] - cy) > radius
            ]
            if len(inside) < 5 or len(outside) < 8:
                continue
            gain = parent - sse(inside) - sse(outside)
            if best is None or gain > best.score:
                best = GaussianSplit(cx, cy, radius, gain)
    if best is None:
        return GaussianSplit(256, 256, 96, 0.0)
    return best


def assign_leaves(nodes: list[Node], leaves: list[Leaf]) -> None:
    for node in nodes:
        x, y = display_point(node)
        for leaf in leaves:
            if leaf.x0 <= x <= leaf.x1 and leaf.y0 <= y <= leaf.y1:
                node.leaf = leaf.index
                break


def path_between(a: Node, b: Node, kind: str) -> str:
    ax, ay = display_point(a)
    bx, by = display_point(b)
    if a.continent != b.continent:
        lift = -42 if abs(ax - bx) > 80 else -20
        return f"M{ax:.2f} {ay:.2f} C{ax + 56:.2f} {ay + lift:.2f} {bx - 56:.2f} {by + lift:.2f} {bx:.2f} {by:.2f}"
    bend = 0.08 if a.continent == b.continent else 0.15
    mx = (ax + bx) / 2 + (by - ay) * bend
    my = (ay + by) / 2 - (bx - ax) * bend
    return f"M{ax:.2f} {ay:.2f} Q{mx:.2f} {my:.2f} {bx:.2f} {by:.2f}"


def line_for_oblique(
    split: ObliqueSplit, bounds: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bounds
    candidates: list[tuple[float, float]] = []
    if abs(split.b) > 1e-9:
        for x in [x0, x1]:
            y = (split.threshold - split.a * x) / split.b
            if y0 <= y <= y1:
                candidates.append((x, y))
    if abs(split.a) > 1e-9:
        for y in [y0, y1]:
            x = (split.threshold - split.b * y) / split.a
            if x0 <= x <= x1:
                candidates.append((x, y))
    if len(candidates) < 2:
        return x0, y0, x1, y1
    return candidates[0][0], candidates[0][1], candidates[-1][0], candidates[-1][1]


def oblique_polygon(
    split: ObliqueSplit, bounds: tuple[float, float, float, float], keep_le: bool
) -> str:
    x0, y0, x1, y1 = bounds
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def inside(point: tuple[float, float]) -> bool:
        value = split.a * point[0] + split.b * point[1]
        return value <= split.threshold if keep_le else value > split.threshold

    def intersect(p: tuple[float, float], q: tuple[float, float]) -> tuple[float, float]:
        pv = split.a * p[0] + split.b * p[1] - split.threshold
        qv = split.a * q[0] + split.b * q[1] - split.threshold
        t = pv / (pv - qv)
        return p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])

    output: list[tuple[float, float]] = []
    for p, q in zip(corners, corners[1:] + corners[:1]):
        p_in = inside(p)
        q_in = inside(q)
        if p_in and q_in:
            output.append(q)
        elif p_in and not q_in:
            output.append(intersect(p, q))
        elif not p_in and q_in:
            output.append(intersect(p, q))
            output.append(q)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in output)


def render_svg(
    nodes: list[Node],
    edges: list[tuple[str, str, str, float]],
    by_key: dict[str, Node],
    leaves: list[Leaf],
    splits: list[Split],
    obliques: list[tuple[ObliqueSplit, tuple[float, float, float, float]]],
    gaussian: GaussianSplit,
) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" role="img" aria-label="CartoBoost splitter route logo">',
        "<defs>",
        '<clipPath id="frame"><rect width="512" height="512" rx="82"/></clipPath>',
        '<filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="3"/></filter>',
        "</defs>",
        '<rect width="512" height="512" rx="82" fill="#0b1220"/>',
        '<g clip-path="url(#frame)">',
    ]
    parts.append(
        f'<circle cx="{gaussian.cx:.2f}" cy="{gaussian.cy:.2f}" r="{gaussian.radius * 1.22:.2f}" fill="#27c2b5" opacity="0.07"/>'
    )
    parts.append(
        f'<circle cx="{gaussian.cx:.2f}" cy="{gaussian.cy:.2f}" r="{gaussian.radius:.2f}" fill="#ffb454" opacity="0.08"/>'
    )
    parts.append(
        f'<circle cx="{gaussian.cx:.2f}" cy="{gaussian.cy:.2f}" r="{gaussian.radius * 0.46:.2f}" fill="#27c2b5" opacity="0.10"/>'
    )
    for split in splits:
        if split.axis == "x":
            parts.append(
                f'<path d="M{split.threshold:.2f} {split.y0:.2f}V{split.y1:.2f}" stroke="#07101e" stroke-width="{5.4 - split.depth * 0.54:.2f}" opacity="0.34"/>'
            )
            parts.append(
                f'<path d="M{split.threshold:.2f} {split.y0:.2f}V{split.y1:.2f}" stroke="#e8eef8" stroke-width="{1.65 - split.depth * 0.18:.2f}" opacity="0.46"/>'
            )
        else:
            parts.append(
                f'<path d="M{split.x0:.2f} {split.threshold:.2f}H{split.x1:.2f}" stroke="#07101e" stroke-width="{5.4 - split.depth * 0.54:.2f}" opacity="0.34"/>'
            )
            parts.append(
                f'<path d="M{split.x0:.2f} {split.threshold:.2f}H{split.x1:.2f}" stroke="#e8eef8" stroke-width="{1.65 - split.depth * 0.18:.2f}" opacity="0.46"/>'
            )
    for oblique, field_bounds in obliques:
        ox0, oy0, ox1, oy1 = line_for_oblique(oblique, field_bounds)
        parts.append(
            f'<path d="M{ox0:.2f} {oy0:.2f}L{ox1:.2f} {oy1:.2f}" stroke="#27c2b5" stroke-width="22" opacity="0.13" stroke-linecap="round"/>'
        )
        parts.append(
            f'<path d="M{ox0:.2f} {oy0:.2f}L{ox1:.2f} {oy1:.2f}" stroke="#07101e" stroke-width="6.6" opacity="0.56" stroke-linecap="round"/>'
        )
        parts.append(
            f'<path d="M{ox0:.2f} {oy0:.2f}L{ox1:.2f} {oy1:.2f}" stroke="#ffb454" stroke-width="2.7" opacity="0.94" stroke-linecap="round"/>'
        )
    parts.append(
        f'<circle cx="{gaussian.cx:.2f}" cy="{gaussian.cy:.2f}" r="{gaussian.radius:.2f}" fill="none" stroke="#27c2b5" stroke-width="3.1" opacity="0.9"/>'
    )
    parts.append(
        f'<circle cx="{gaussian.cx:.2f}" cy="{gaussian.cy:.2f}" r="{gaussian.radius * 0.46:.2f}" fill="none" stroke="#27c2b5" stroke-width="1.8" opacity="0.62"/>'
    )
    for a_key, b_key, kind, _ in sorted(
        edges, key=lambda edge: by_key[edge[0]].continent != by_key[edge[1]].continent
    ):
        a = by_key[a_key]
        b = by_key[b_key]
        cross_continent = a.continent != b.continent
        color = "#d7f2ff" if cross_continent else TREE_COLORS[a.leaf % len(TREE_COLORS)]
        width = 3.1 if cross_continent else 1.4 + 2.2 * max(a.risk, b.risk)
        dash = ' stroke-dasharray="8 7"' if cross_continent else ""
        parts.append(
            f'<path d="{path_between(a, b, kind)}" fill="none" stroke="#07101e" stroke-width="{width + 2.2:.2f}" opacity="0.52" stroke-linecap="round"{dash}/>'
        )
        parts.append(
            f'<path d="{path_between(a, b, kind)}" fill="none" stroke="{color}" stroke-width="{width:.2f}" opacity="{0.84 if cross_continent else 0.68}" stroke-linecap="round"{dash}/>'
        )
    for node in sorted(nodes, key=lambda item: item.risk):
        x, y = display_point(node)
        color = TREE_COLORS[node.leaf % len(TREE_COLORS)]
        r = 3.2 + node.risk * 5.2
        if any(
            by_key[other].continent != node.continent
            for edge in edges
            for other in edge[:2]
            if node.key in edge[:2] and other != node.key
        ):
            parts.append(
                f'<rect x="{x - r:.2f}" y="{y - r:.2f}" width="{2 * r:.2f}" height="{2 * r:.2f}" rx="2.4" fill="#07101e" opacity="0.9"/>'
            )
            parts.append(
                f'<rect x="{x - r * 0.52:.2f}" y="{y - r * 0.52:.2f}" width="{r * 1.04:.2f}" height="{r * 1.04:.2f}" rx="1.4" fill="#d7f2ff"/>'
            )
        else:
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="#07101e" opacity="0.92"/>'
            )
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r * 0.48:.2f}" fill="{color}"/>')
    parts.extend(
        [
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    ensure_inputs()
    nodes = parse_nodes()
    edges, by_key = graph_edges(nodes)
    compute_risk(nodes, edges, by_key)
    leaves: list[Leaf] = []
    splits: list[Split] = []
    fit_tree(nodes, (-28, 18, 540, 484), 0, 3, leaves, splits)
    assign_leaves(nodes, leaves)
    kalimdor = [node for node in nodes if node.continent == "Kalimdor"]
    eastern = [node for node in nodes if node.continent == "Eastern Kingdoms"]
    obliques = [
        (fit_oblique_split(kalimdor), (LEFT_X, PANEL_Y, LEFT_X + PANEL_W, PANEL_Y + PANEL_H)),
        (fit_oblique_split(eastern), (RIGHT_X, PANEL_Y, RIGHT_X + PANEL_W, PANEL_Y + PANEL_H)),
    ]
    gaussian = fit_gaussian_split(nodes)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_svg(nodes, edges, by_key, leaves, splits, obliques, gaussian))
    print(f"wrote {OUT}")
    print(f"nodes={len(nodes)} edges={len(edges)} leaves={len(leaves)} splits={len(splits)}")


if __name__ == "__main__":
    main()
