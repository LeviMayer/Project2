import os
import json
import csv
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


# ----------------------------
# Config
# ----------------------------
@dataclass
class GenConfig:
    out_dir: str = "out_lineex_v3"
    n_samples: int = 15000
    img_size: int = 224
    dpi: int = 100

    # data
    n_lines_min: int = 1
    n_lines_max: int = 1
    n_points_min: int = 5
    n_points_max: int = 12

    # value ranges
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = 0.0
    y_max: float = 100.0

    # visual randomness
    grid_prob: float = 0.75
    legend_prob: float = 0.25
    markers_prob: float = 0.55
    noise_prob: float = 0.08
    thick_prob: float = 0.35
    value_labels_prob: float = 0.45

    # labels / title
    title_prob: float = 0.9
    axis_label_prob: float = 0.85

    # keep text cleaner than before
    text_noise_prob: float = 0.0
    text_jitter_px: float = 0.0
    tick_rotation_prob: float = 0.08
    legend_jitter_prob: float = 0.0

    # masking
    make_masked: bool = True
    mask_rect_prob: float = 1.0
    mask_rect_min_area: float = 0.08
    mask_rect_max_area: float = 0.22

    seed: int = 42


# ----------------------------
# Text helpers
# ----------------------------
_TITLE_PREFIX = [
    "Monthly", "Quarterly", "Annual", "Weekly", "Daily", "6-Month", "12-Month"
]
_TITLE_SUBJECTS = [
    "sales report", "forecast", "utilization", "performance", "trend",
    "growth", "revenue", "cost analysis", "speed", "latency", "accuracy"
]
_AXIS_WORDS_X = [
    "Time", "Month", "Quarter", "Epoch", "Step", "Distance", "Voltage", "Iteration"
]
_AXIS_WORDS_Y = [
    "Sales", "Revenue", "Cost", "Speed", "Latency", "Utilization", "Score", "Value", "Units"
]
_UNITS = ["USD", "%", "ms", "s", "km/h", "items", "units", "GB", "MB", "kWh"]

_MONTH_LABEL_SETS = [
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"],
    ["JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
    ["OCT\n2019", "NOV", "DEC", "JAN\n2020", "FEB", "MAR"],
    ["APR", "MAY", "JUN", "JUL", "AUG", "SEP"],
]


def random_title() -> str:
    if random.random() < 0.45:
        return f"{random.choice(_TITLE_PREFIX)} {random.choice(_TITLE_SUBJECTS)}"
    return f"{random.choice(_TITLE_PREFIX)} {random.choice(_TITLE_SUBJECTS)} and {random.choice(['forecast', 'trend', 'projection'])}"


def random_axis_label(kind: str) -> str:
    if kind == "x":
        base = random.choice(_AXIS_WORDS_X)
        if random.random() < 0.6:
            return base
        return f"{base} ({random.choice(['days', 'months', 'steps', 'units'])})"
    else:
        base = random.choice(_AXIS_WORDS_Y)
        if random.random() < 0.6:
            return base
        return f"{base} ({random.choice(_UNITS)})"


# ----------------------------
# X-axis generation
# ----------------------------
def make_x_axis(n_pts: int, cfg: GenConfig) -> Tuple[np.ndarray, List[str], str]:
    """
    Returns:
      xs_numeric: numeric positions for plotting
      tick_labels: labels shown on axis
      axis_mode: one of ['numeric', 'months', 'categories']
    """
    mode = random.choices(
        ["numeric", "months", "categories"],
        weights=[0.45, 0.35, 0.20],
        k=1
    )[0]

    if mode == "months":
        labels = random.choice(_MONTH_LABEL_SETS)
        if n_pts != len(labels):
            n_pts = len(labels)
        xs = np.arange(n_pts, dtype=np.float32)
        return xs, labels[:n_pts], mode

    if mode == "categories":
        labels = [f"C{i+1}" for i in range(n_pts)]
        xs = np.arange(n_pts, dtype=np.float32)
        return xs, labels, mode

    # numeric
    xs = np.linspace(cfg.x_min, cfg.x_max, n_pts, dtype=np.float32)
    labels = [f"{x:.2f}" if random.random() < 0.5 else f"{x:.1f}" for x in xs]
    return xs, labels, mode


# ----------------------------
# Curve generation
# ----------------------------
def random_curve(xs: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    n = len(xs)
    span = y_max - y_min

    style = random.choices(
        ["business_piecewise", "monotone", "stair", "random_walk", "smooth"],
        weights=[0.42, 0.18, 0.15, 0.18, 0.07],
        k=1
    )[0]

    def clip(y):
        return np.clip(y, y_min, y_max)

    if style == "business_piecewise":
        k = random.randint(max(4, n // 3), min(n, max(5, n)))
        knot_idx = np.sort(np.random.choice(np.arange(n), size=k, replace=False))
        if knot_idx[0] != 0:
            knot_idx = np.insert(knot_idx, 0, 0)
        if knot_idx[-1] != n - 1:
            knot_idx = np.append(knot_idx, n - 1)

        values = np.random.uniform(y_min + 0.05 * span, y_max - 0.05 * span, size=len(knot_idx))
        y = np.interp(np.arange(n), knot_idx, values)

        if random.random() < 0.55:
            q = random.choice([1, 2, 2.5, 5, 10])
            y = np.round(y / q) * q

        return clip(y)

    if style == "monotone":
        direction = random.choice([1, -1])
        steps = np.abs(np.random.normal(0.08 * span, 0.04 * span, size=n))
        steps[np.random.rand(n) < 0.25] = 0
        y = np.zeros(n, dtype=np.float32)
        y[0] = random.uniform(y_min + 0.1 * span, y_max - 0.1 * span)
        for i in range(1, n):
            y[i] = y[i - 1] + direction * steps[i]
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        y = y_min + y * span
        if random.random() < 0.6:
            q = random.choice([1, 2, 5])
            y = np.round(y / q) * q
        return clip(y)

    if style == "stair":
        n_steps = random.randint(3, min(max(4, n - 1), 8))
        cuts = np.sort(np.random.choice(np.arange(1, n), size=n_steps - 1, replace=False))
        cuts = np.concatenate([[0], cuts, [n]])
        levels = np.random.uniform(y_min + 0.05 * span, y_max - 0.05 * span, size=len(cuts) - 1)
        y = np.zeros(n, dtype=np.float32)
        for i in range(len(levels)):
            y[cuts[i]:cuts[i + 1]] = levels[i]
        if random.random() < 0.5:
            y = np.maximum.accumulate(y)
        if random.random() < 0.7:
            q = random.choice([1, 2, 5])
            y = np.round(y / q) * q
        return clip(y)

    if style == "random_walk":
        y = np.zeros(n, dtype=np.float32)
        y[0] = random.uniform(y_min + 0.1 * span, y_max - 0.1 * span)
        for i in range(1, n):
            y[i] = y[i - 1] + np.random.normal(0, 0.09 * span)
        if random.random() < 0.45:
            q = random.choice([1, 2, 5])
            y = np.round(y / q) * q
        return clip(y)

    t = np.linspace(0, 2 * np.pi, n)
    base = random.uniform(y_min + 0.25 * span, y_max - 0.25 * span)
    amp = random.uniform(0.08 * span, 0.22 * span)
    y = base + amp * np.sin(t + random.uniform(0, 2 * np.pi))
    if random.random() < 0.3:
        y += np.random.normal(0, 0.03 * span, size=n)
    return clip(y)


# ----------------------------
# Noise & masking
# ----------------------------
def add_pixel_noise(pil_img: Image.Image, strength: float = 5.0) -> Image.Image:
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, strength, size=arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def mask_random_rectangle(pil_img: Image.Image, area_min: float, area_max: float) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    area = random.uniform(area_min, area_max) * W * H
    rw = int(math.sqrt(area) * random.uniform(0.8, 1.2))
    rh = int(area / max(rw, 1))

    rw = max(10, min(rw, W - 1))
    rh = max(10, min(rh, H - 1))

    x0 = random.randint(0, W - rw)
    y0 = random.randint(0, H - rh)

    draw.rectangle([x0, y0, x0 + rw, y0 + rh], fill=(60, 60, 60))
    return img


# ----------------------------
# Styling helpers
# ----------------------------
_BUSINESS_COLORS = [
    "#000000", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#17becf", "#ff7f0e"
]


def pick_line_color(n_lines: int) -> List[str]:
    if n_lines == 1:
        if random.random() < 0.35:
            return ["#000000"]
        return [random.choice(_BUSINESS_COLORS)]
    colors = random.sample(_BUSINESS_COLORS, k=min(n_lines, len(_BUSINESS_COLORS)))
    return colors


def choose_chart_style():
    style_name = random.choices(
        ["clean_business", "classic_line", "marker_line", "report_style"],
        weights=[0.28, 0.24, 0.24, 0.24],
        k=1
    )[0]

    style = {
        "name": style_name,
        "show_markers": False,
        "show_value_labels": False,
        "linewidth": 2.0,
        "marker": None,
        "marker_size": 4,
        "monochrome": False,
    }

    if style_name == "clean_business":
        style["linewidth"] = random.choice([2.2, 2.5, 2.8])
        style["show_markers"] = random.random() < 0.5
        style["marker"] = random.choice(["o", "s"])
        style["show_value_labels"] = random.random() < 0.35

    elif style_name == "classic_line":
        style["linewidth"] = random.choice([1.5, 1.8, 2.0])
        style["show_markers"] = False
        style["show_value_labels"] = False

    elif style_name == "marker_line":
        style["linewidth"] = random.choice([1.8, 2.2, 2.5])
        style["show_markers"] = True
        style["marker"] = random.choice(["o", "s", "^", "D"])
        style["marker_size"] = random.choice([4, 5, 6])
        style["show_value_labels"] = random.random() < 0.55

    elif style_name == "report_style":
        style["linewidth"] = random.choice([2.5, 3.0, 3.2])
        style["show_markers"] = True
        style["marker"] = "o"
        style["marker_size"] = random.choice([5, 6, 7])
        style["show_value_labels"] = True
        style["monochrome"] = random.random() < 0.6

    return style


# ----------------------------
# Pixel metadata extraction
# ----------------------------
def extract_pixel_metadata(fig, ax, xs: np.ndarray, ys_lines: List[np.ndarray]):
    """
    Extract exact plotted pixel coordinates from Matplotlib canvas.

    Returns:
        plot_bbox_px: [x0, y0, w, h] in image coordinates (origin top-left)
        points_px: list of dicts with exact plotted point positions
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    canvas_w, canvas_h = fig.canvas.get_width_height()

    bbox = ax.get_window_extent(renderer=renderer)
    x0 = float(bbox.x0)
    y0 = float(canvas_h - bbox.y1)
    w = float(bbox.width)
    h = float(bbox.height)

    points_px = []
    for line_id, ys in enumerate(ys_lines):
        xy = np.column_stack([xs, ys])
        disp = ax.transData.transform(xy)
        for px, py in disp:
            points_px.append({
                "line_id": int(line_id),
                "x": float(px),
                "y": float(canvas_h - py),
            })

    return [x0, y0, w, h], points_px


# ----------------------------
# Render chart
# ----------------------------
def render_chart(
    xs: np.ndarray,
    x_labels: List[str],
    x_mode: str,
    ys_lines: List[np.ndarray],
    cfg: GenConfig,
    out_png: str,
    show_grid: bool,
    show_legend: bool,
):
    fig = plt.figure(figsize=(cfg.img_size / cfg.dpi, cfg.img_size / cfg.dpi), dpi=cfg.dpi)

    # more natural layout
    ax = fig.add_axes([0.14, 0.16, 0.76, 0.68])

    style = choose_chart_style()
    colors = pick_line_color(len(ys_lines))
    if style["monochrome"]:
        colors = ["#000000" for _ in ys_lines]

    # background / spines
    bg = "#ffffff" if random.random() < 0.8 else "#f8f8f8"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if random.random() < 0.8:
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")

    if show_grid:
        ax.grid(True, axis="y" if random.random() < 0.65 else "both", linewidth=0.6, alpha=0.45, color="#bcbcbc")
    else:
        ax.grid(False)

    ax.set_ylim(cfg.y_min, cfg.y_max)
    ax.set_xlim(xs[0], xs[-1] if len(xs) > 1 else xs[0] + 1e-3)

    # ticks
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=random.choice([6, 7, 8]), color="#666666")
    if random.random() < cfg.tick_rotation_prob:
        for t in ax.get_xticklabels():
            t.set_rotation(random.choice([10, 15, 20, -10]))

    y_ticks = np.linspace(cfg.y_min, cfg.y_max, random.choice([5, 6, 7]))
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelsize=random.choice([6, 7, 8]), colors="#666666")
    ax.tick_params(axis="x", colors="#777777")

    # labels/title
    if random.random() < cfg.axis_label_prob:
        ax.set_xlabel(random_axis_label("x"), fontsize=random.choice([8, 9]), color="#666666")
    if random.random() < cfg.axis_label_prob:
        ax.set_ylabel(random_axis_label("y"), fontsize=random.choice([8, 9]), color="#666666")
    elif random.random() < 0.2:
        ax.text(
            0.0, 1.03,
            f"IN {random.choice(['THOUSANDS', 'MILLIONS', 'UNITS'])} ({random.choice(_UNITS)})",
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=7,
            color="#777777"
        )

    if random.random() < cfg.title_prob:
        ax.set_title(random_title(), fontsize=random.choice([9, 10, 11]), pad=random.choice([6, 8, 10]), color="#555555")

    # plot lines
    for i, ys in enumerate(ys_lines):
        kwargs = dict(
            linewidth=style["linewidth"],
            color=colors[i],
            alpha=0.98,
        )
        if style["show_markers"]:
            kwargs["marker"] = style["marker"]
            kwargs["markersize"] = style["marker_size"]
            kwargs["markerfacecolor"] = colors[i]
            kwargs["markeredgecolor"] = colors[i]

        ax.plot(xs, ys, **kwargs)

        if style["show_value_labels"] or (random.random() < cfg.value_labels_prob and len(ys_lines) == 1):
            for x, y in zip(xs, ys):
                if random.random() < 0.75:
                    va = "bottom" if random.random() < 0.5 else "top"
                    yoff = 2.5 if va == "bottom" else -2.5
                    ax.text(
                        x, y + yoff,
                        f"{y:.1f}",
                        fontsize=5 if cfg.img_size <= 224 else 6,
                        color="#444444",
                        ha="center",
                        va=va
                    )

    # legend
    if show_legend and len(ys_lines) > 1:
        labels = [f"Series {i+1}" for i in range(len(ys_lines))]
        loc = random.choice(["upper left", "upper right", "lower left", "lower right"])
        ax.legend(labels, loc=loc, fontsize=6, framealpha=0.9)

    plot_bbox_px, points_px = extract_pixel_metadata(fig, ax, xs, ys_lines)

    fig.savefig(out_png, dpi=cfg.dpi, bbox_inches=None)
    plt.close(fig)

    # Keep exact geometry; no extra resizing
    Image.open(out_png).convert("RGB").save(out_png)

    return {
        "plot_bbox_px": plot_bbox_px,
        "points_px": points_px,
    }


# ----------------------------
# Main generator
# ----------------------------
def generate(cfg: GenConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    img_dir = os.path.join(cfg.out_dir, "images")
    lab_dir = os.path.join(cfg.out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    manifest = os.path.join(cfg.out_dir, "manifest.jsonl")

    with open(manifest, "w") as mf:
        for idx in range(cfg.n_samples):
            sid = f"chart_{idx:06d}"

            n_lines = random.randint(cfg.n_lines_min, cfg.n_lines_max)
            n_pts = random.randint(cfg.n_points_min, cfg.n_points_max)

            xs, x_labels, x_mode = make_x_axis(n_pts, cfg)
            ys_lines = [random_curve(xs, cfg.y_min, cfg.y_max) for _ in range(n_lines)]

            full = os.path.join(img_dir, f"{sid}_full.png")
            masked = os.path.join(img_dir, f"{sid}_masked.png")
            csv_p = os.path.join(lab_dir, f"{sid}.csv")

            render_meta = render_chart(
                xs=xs,
                x_labels=x_labels,
                x_mode=x_mode,
                ys_lines=ys_lines,
                cfg=cfg,
                out_png=full,
                show_grid=(random.random() < cfg.grid_prob),
                show_legend=(random.random() < cfg.legend_prob),
            )

            # optional mild pixel noise
            if random.random() < cfg.noise_prob:
                img = Image.open(full).convert("RGB")
                img = add_pixel_noise(img, strength=random.uniform(1.0, 5.0))
                img.save(full)

            # masked version
            if cfg.make_masked and random.random() < cfg.mask_rect_prob:
                img = Image.open(full).convert("RGB")
                img = mask_random_rectangle(img, cfg.mask_rect_min_area, cfg.mask_rect_max_area)
                img.save(masked)
            else:
                masked = None

            # write CSV labels
            with open(csv_p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "line_id"])
                for li, ys in enumerate(ys_lines):
                    for x, y in zip(xs, ys):
                        w.writerow([float(x), float(y), li])

            mf.write(json.dumps({
                "id": sid,
                "full": os.path.relpath(full, cfg.out_dir),
                "masked": os.path.relpath(masked, cfg.out_dir) if masked else None,
                "csv": os.path.relpath(csv_p, cfg.out_dir),
                "n_lines": n_lines,
                "n_points": n_pts,
                "x_mode": x_mode,
                "plot_bbox_px": render_meta["plot_bbox_px"],
                "points_px": render_meta["points_px"],
            }) + "\n")

            if (idx + 1) % 500 == 0:
                print(f"[{idx+1}/{cfg.n_samples}] generated")

    print("Done.")
    print("Manifest:", manifest)


if __name__ == "__main__":
    cfg = GenConfig(
        out_dir="out_lineex_v2",
        n_samples=20000,
        img_size=224,
        make_masked=True,
        seed=42
    )
    generate(cfg)