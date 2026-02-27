import os, json, csv, math, random
from dataclasses import dataclass
from typing import List, Optional

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
    out_dir: str = "out"
    n_samples: int = 20000
    img_size: int = 224
    dpi: int = 100

    # data
    n_lines_min: int = 1
    n_lines_max: int = 1
    n_points_min: int = 20
    n_points_max: int = 60

    # value ranges
    x_min: float = 0.0
    x_max: float = 10.0
    y_min: float = 0.0
    y_max: float = 100.0

    # visual randomness
    grid_prob: float = 0.9
    legend_prob: float = 0.5
    markers_prob: float = 0.3
    noise_prob: float = 0.25
    thick_prob: float = 0.3

    # labels / title
    title_prob: float = 0.75
    axis_label_prob: float = 0.9

    # text noise / jitter
    text_noise_prob: float = 0.9
    text_jitter_px: float = 2.5      # shift in pixels (approx)
    tick_rotation_prob: float = 0.2  # sometimes rotate tick labels a bit
    legend_jitter_prob: float = 0.8  # legend slightly moved

    # masking
    make_masked: bool = True
    mask_rect_prob: float = 1.0
    mask_rect_min_area: float = 0.08
    mask_rect_max_area: float = 0.25

    seed: int = 42


# ----------------------------
# Random text helpers
# ----------------------------
_TITLE_WORDS = [
    "Revenue", "Latency", "Accuracy", "Loss", "Sales", "Temperature", "Pressure", "Demand",
    "Growth", "Utilization", "Throughput", "Voltage", "Speed", "Energy", "Cost", "Index"
]
_AXIS_WORDS = [
    "Time", "Epoch", "Step", "Day", "Month", "Iteration", "Sample", "Distance", "Angle",
    "Frequency", "Voltage", "Count", "Value", "Score", "Rate"
]
_UNITS = ["ms", "s", "min", "°C", "kPa", "%", "USD", "EUR", "GB", "MB", "kWh", "units"]

def random_title() -> str:
    a = random.choice(_TITLE_WORDS)
    b = random.choice(["over", "vs", "by", "across", "per"])
    c = random.choice(_AXIS_WORDS)
    # sometimes short, sometimes longer
    if random.random() < 0.35:
        return f"{a}"
    if random.random() < 0.35:
        return f"{a} {b} {c}"
    return f"{a} {b} {c} ({random.choice(['Experiment', 'Run', 'Trial', 'Dataset'])} {random.randint(1, 9)})"

def random_axis_label(kind: str) -> str:
    if kind == "x":
        base = random.choice(_AXIS_WORDS)
        if random.random() < 0.55:
            return base
        return f"{base} ({random.choice(_UNITS)})"
    else:
        base = random.choice(_TITLE_WORDS + _AXIS_WORDS)
        if random.random() < 0.55:
            return base
        return f"{base} ({random.choice(_UNITS)})"


# ----------------------------
# Curve generation (jagged/monotone/stair/etc.)
# ----------------------------
def random_curve(xs: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    n = len(xs)
    span = y_max - y_min

    style = random.choices(
        ["zigzag", "monotone", "stair", "random_walk", "smooth_rare"],
        weights=[0.40, 0.25, 0.20, 0.13, 0.02],
        k=1
    )[0]

    def clip(y):
        return np.clip(y, y_min, y_max)

    # ZIGZAG / SHARP
    if style == "zigzag":
        k = random.randint(6, 14)
        knots = np.sort(np.random.choice(np.arange(n), size=k, replace=False))
        if knots[0] != 0:
            knots = np.insert(knots, 0, 0)
        if knots[-1] != n - 1:
            knots = np.append(knots, n - 1)

        values = np.random.uniform(y_min, y_max, size=len(knots))
        y = np.interp(np.arange(n), knots, values)

        for _ in range(random.randint(1, 3)):
            j = random.randint(3, n - 3)
            y[j:] += random.uniform(-0.3 * span, 0.3 * span)

        if random.random() < 0.7:
            q = random.choice([0.5, 1, 2, 5])
            y = np.round(y / q) * q

        return clip(y)

    # MONOTONE
    if style == "monotone":
        direction = random.choice([1, -1])
        steps = np.abs(np.random.normal(0.04 * span, 0.02 * span, size=n))
        steps[np.random.rand(n) < 0.4] = 0

        y = np.zeros(n)
        y[0] = random.uniform(y_min, y_max)
        for i in range(1, n):
            y[i] = y[i - 1] + direction * steps[i]

        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        y = y_min + y * span
        return clip(y)

    # STAIR / STEP
    if style == "stair":
        n_steps = random.randint(4, min(12, n))
        cuts = np.sort(np.random.choice(np.arange(1, n), size=n_steps - 1, replace=False))
        cuts = np.concatenate([[0], cuts, [n]])

        levels = np.random.uniform(y_min, y_max, size=len(cuts) - 1)
        y = np.zeros(n)
        for i in range(len(levels)):
            y[cuts[i]:cuts[i + 1]] = levels[i]

        if random.random() < 0.5:
            y = np.maximum.accumulate(y)

        return clip(y)

    # RANDOM WALK
    if style == "random_walk":
        y = np.zeros(n)
        y[0] = random.uniform(y_min, y_max)
        for i in range(1, n):
            y[i] = y[i - 1] + np.random.normal(0, 0.05 * span)
        return clip(y)

    # VERY RARE SMOOTH
    t = np.linspace(0, 2 * np.pi, n)
    base = random.uniform(y_min + 0.3 * span, y_max - 0.3 * span)
    amp = random.uniform(0.05 * span, 0.2 * span)
    return clip(base + amp * np.sin(t + random.uniform(0, 2 * np.pi)))


# ----------------------------
# Noise & masking
# ----------------------------
def add_pixel_noise(pil_img: Image.Image, strength: float = 8.0) -> Image.Image:
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
# Text noise / jitter helpers (Matplotlib)
# ----------------------------
def _jitter_text_obj(t, cfg: GenConfig):
    """Jitter a matplotlib Text object slightly."""
    if t is None:
        return
    dx = random.uniform(-cfg.text_jitter_px, cfg.text_jitter_px)
    dy = random.uniform(-cfg.text_jitter_px, cfg.text_jitter_px)
    # text positions are in data coords by default; use offset points via transform trick:
    # easiest: change transform to offset points from current position
    x, y = t.get_position()
    trans = t.get_transform()
    # apply offset in points
    t.set_transform(trans + matplotlib.transforms.ScaledTranslation(dx/72, dy/72, plt.gcf().dpi_scale_trans))

def _maybe_jitter_ticklabels(ax, cfg: GenConfig):
    for t in (ax.get_xticklabels() + ax.get_yticklabels()):
        if random.random() < 0.85:
            _jitter_text_obj(t, cfg)
        if random.random() < 0.25:
            t.set_fontsize(max(5, int(t.get_fontsize() + random.choice([-1, 0, 1]))))
        if random.random() < cfg.tick_rotation_prob:
            t.set_rotation(random.choice([-20, -15, -10, 10, 15, 20]))

def _maybe_jitter_legend(leg, cfg: GenConfig):
    if leg is None:
        return
    if random.random() < cfg.legend_jitter_prob:
        # small bbox shift in axes coordinates
        dx = random.uniform(-0.03, 0.03)
        dy = random.uniform(-0.03, 0.03)
        loc = leg._loc  # keep chosen loc
        bbox = leg.get_bbox_to_anchor()
        if bbox is None:
            bbox = matplotlib.transforms.Bbox.from_bounds(0, 0, 1, 1)
        # set bbox_to_anchor in axes coords
        leg.set_bbox_to_anchor((dx, dy, 1, 1), transform=leg.axes.transAxes)
        leg._loc = loc
    # jitter legend text
    for t in leg.get_texts():
        if random.random() < 0.9:
            _jitter_text_obj(t, cfg)
        if random.random() < 0.25:
            t.set_fontsize(max(5, int(t.get_fontsize() + random.choice([-1, 0, 1]))))


# ----------------------------
# Render chart
# ----------------------------
def render_chart(
    xs: np.ndarray,
    ys_lines: List[np.ndarray],
    cfg: GenConfig,
    out_png: str,
    show_grid: bool,
    show_legend: bool,
    show_markers: bool,
    thick_lines: bool
):
    fig = plt.figure(figsize=(cfg.img_size / cfg.dpi, cfg.img_size / cfg.dpi), dpi=cfg.dpi)
    ax = fig.add_axes([0.12, 0.14, 0.82, 0.78])  # a bit more bottom margin for x-label

    if show_grid:
        ax.grid(True, linewidth=0.6, alpha=0.7)

    ax.set_xlim(cfg.x_min, cfg.x_max)
    ax.set_ylim(cfg.y_min, cfg.y_max)

    ax.set_xticks(np.linspace(cfg.x_min, cfg.x_max, random.choice([5, 6, 7, 8])))
    ax.set_yticks(np.linspace(cfg.y_min, cfg.y_max, random.choice([5, 6, 7, 8])))

    # labels/title
    if random.random() < cfg.axis_label_prob:
        ax.set_xlabel(random_axis_label("x"), fontsize=random.choice([7, 8, 9]))
    if random.random() < cfg.axis_label_prob:
        ax.set_ylabel(random_axis_label("y"), fontsize=random.choice([7, 8, 9]))
    title_text = None
    if random.random() < cfg.title_prob:
        title_text = ax.set_title(random_title(), fontsize=random.choice([8, 9, 10]), pad=random.choice([2, 3, 4]))

    lw = random.choice([1.2, 1.5, 2.0]) if not thick_lines else random.choice([2.5, 3.0, 3.5])

    for i, ys in enumerate(ys_lines):
        if show_markers and random.random() < 0.7:
            ax.plot(xs, ys, marker=random.choice(["o", "s", "^"]), markersize=3, linewidth=lw)
        else:
            ax.plot(xs, ys, linewidth=lw)

    leg = None
    if show_legend and len(ys_lines) > 1:
        labels = [f"Series {i+1}" for i in range(len(ys_lines))]
        loc = random.choice(["upper left", "upper right", "lower left", "lower right"])
        leg = ax.legend(labels, loc=loc, fontsize=6, framealpha=0.85)

    ax.tick_params(labelsize=6)

    # ---- text-noise: jitter ticks/legend/title/axislabels ----
    if random.random() < cfg.text_noise_prob:
        _maybe_jitter_ticklabels(ax, cfg)
        if title_text is not None and random.random() < 0.9:
            _jitter_text_obj(title_text, cfg)
        if ax.xaxis.label is not None and random.random() < 0.9:
            _jitter_text_obj(ax.xaxis.label, cfg)
        if ax.yaxis.label is not None and random.random() < 0.9:
            _jitter_text_obj(ax.yaxis.label, cfg)
        _maybe_jitter_legend(leg, cfg)

    fig.savefig(out_png, dpi=cfg.dpi)
    plt.close(fig)

    # enforce exact size & RGB
    Image.open(out_png).convert("RGB").resize((cfg.img_size, cfg.img_size), Image.BICUBIC).save(out_png)


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

            xs = np.linspace(cfg.x_min, cfg.x_max, n_pts)
            ys_lines = [random_curve(xs, cfg.y_min, cfg.y_max) for _ in range(n_lines)]

            full = os.path.join(img_dir, f"{sid}_full.png")
            masked = os.path.join(img_dir, f"{sid}_masked.png")
            csv_p = os.path.join(lab_dir, f"{sid}.csv")

            render_chart(
                xs, ys_lines, cfg, full,
                show_grid=(random.random() < cfg.grid_prob),
                show_legend=(random.random() < cfg.legend_prob),
                show_markers=(random.random() < cfg.markers_prob),
                thick_lines=(random.random() < cfg.thick_prob),
            )

            # optional pixel noise
            if random.random() < cfg.noise_prob:
                img = Image.open(full).convert("RGB")
                img = add_pixel_noise(img, strength=random.uniform(2.0, 10.0))
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
                "n_points": n_pts
            }) + "\n")

            if (idx + 1) % 500 == 0:
                print(f"[{idx+1}/{cfg.n_samples}] generated")

    print("Done.")
    print("Manifest:", manifest)


if __name__ == "__main__":
    cfg = GenConfig(
        out_dir="out_lineex",
        n_samples=15000,
        img_size=224,
        make_masked=True,
        seed=42
    )
    generate(cfg)
