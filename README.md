# Project2 — Line Chart Data Extraction with JEPA

Recovering underlying data points from line chart images using Deep Learning and frozen Vision Transformers.

---

## Core Idea

**Goal:** Extract data points from line chart images  
**Method:** Frozen JEPA/ViT encoder + slot-based point decoder  
**Key Innovation:** Dedicated per-slot prediction instead of single point heatmap

```
Chart Image
    ↓
  [Frozen JEPA Encoder]
    ↓
  [Shared Decoder Trunk]
    ├─→ Line Heatmap Head
    └─→ Slot-Based Point Head
    ↓
  [Peak Extraction]
    ↓
  [Recovered (x,y) Data Points]
```

---

## Why Slot-Based Point Heatmaps?

### The Problem
Single point heatmaps fail because:
- Model learns "important curve regions" instead of discrete points
- Results become too line-like
- Hard to extract exact point locations

### The Solution  
**Each point gets its own channel:**

| Slot | Represents |
|------|---|
| Slot 0 | First point |
| Slot 1 | Second point |
| Slot 2 | Third point |
| ... | ... |
| Slot K-1 | Last point slot |

### Benefits
- Clear separation of concerns  
- Each slot individually supervised
- Empty slots can be masked
- Much higher precision + easier extraction
- **Result:** Significantly better than earlier single-heatmap approach

---

## Architecture

### Synthetic Dataset (v3)

Generated datasets follow this structure:

```
out_lineex_v3/
├── images/                    # Chart images
├── labels/                    # Ground truth labels
├── manifest.jsonl             # Full metadata
├── train_manifest.jsonl       # Training split
└── val_manifest.jsonl         # Validation split
```

**Key v3 improvements:**
- ✓ Exact pixel point coordinates (`points_px`)
- ✓ Optional plot bounding boxes (`plot_bbox_px`)
- ✓ Avoids inaccurate CSV-to-pixel reprojections

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Generator** | `gen_linecharts3.py` | Creates synthetic charts + exact pixel labels |
| **Dataset** | `app/lineheatmap2/dataset.py` | Builds line heatmaps + slot point heatmaps |
| **Model** | `app/lineheatmap2/model.py` | Frozen encoder + decoder with dual heads |
| **Training** | `app/lineheatmap2/train.py` | Trains line + slot-based point prediction |
| **Prediction** | `app/lineheatmap2/predict_heatmap.py` | Inference + visualization |
| **Evaluation** | `evals/eval_point_slots.py` | Per-slot metrics (precision/recall/F1) |

---

## Quick Start

### Step 1: Generate Synthetic Charts

```bash
python gen_linecharts3.py
```

Creates `out_lineex_v3/` with:
- Synthetic line chart images
- CSV labels
- Exact pixel point metadata

**Config:** Edit `GenConfig` at the bottom of `gen_linecharts3.py` to change:
- Number of samples
- Output directory
- Image size
- Chart styles

### Step 2: Configure Training

Edit `configs/lineheatmap/lineheatmap_vitl16v2.yaml`:

```yaml
data:
  root_dir: /path/to/out_lineex_v3
  train_manifest: train_manifest.jsonl
  val_manifest: val_manifest.jsonl
  point_sigma: 1.0            # Gaussian width for point targets
  max_points: 16              # Maximum point slots

model:
  point_slots: 16             # Number of point channels

train:
  point_loss_weight: 50.0     # Overall point supervision weight
  point_bce_weight: 1.0       # BCE component
  point_mse_weight: 1.0       # MSE component
```

### Step 3: Train Model

```bash
python -u -m app.lineheatmap2.train \
  --config configs/lineheatmap/lineheatmap_vitl16v2.yaml
```

**Expected output:**
```
DEBUG line_heatmap: [B, 1, 224, 224]
DEBUG point_heatmaps: [B, 16, 224, 224]
DEBUG point_valid_mask: [B, 16]
DEBUG line_logits: [B, 1, 224, 224]
DEBUG point_logits: [B, 16, 224, 224]
```

**Checkpoints saved to:** `outputs/lineheatmap_vitl16_slots/`

### Step 4: Predict on Validation Set

```bash
python -u -m app.lineheatmap2.predict_heatmap \
  --config configs/lineheatmap/lineheatmap_vitl16v2.yaml \
  --checkpoint outputs/lineheatmap_vitl16_slots/best.pth \
  --num_samples 200 \
  --start_index 0 \
  --out_dir outputs/predictions_slots_v3
```

**Output per sample:**
- `chart_000123.png` — Visualization
- `chart_000123_line.npy` — Line heatmap
- `chart_000123_point_slots.npy` — Per-slot points
- `chart_000123_point_agg.npy` — Aggregated points

### Step 5: Evaluate Point Predictions

```bash
python evals/eval_point_slots.py \
  --manifest out_lineex_v3/val_manifest.jsonl \
  --pred_dir outputs/predictions_slots_v3 \
  --x_min 0 --x_max 10 \
  --y_min 0 --y_max 100 \
  --slot_threshold 0.05 \
  --dist_thresh 10.0 \
  --line_id 0 \
  --save_json outputs/predictions_slots_v3/eval_point_slots.json
```

**Metrics reported:**
- TP / FP / FN
- Precision / Recall / F1
- Mean distance to GT points

### Step 6: Sweep Threshold Settings

Fine-tune the slot threshold for your use case:

```bash
# More conservative (higher recall)
python evals/eval_point_slots.py ... --slot_threshold 0.03 --save_json eval_t003.json

# Balanced (default)
python evals/eval_point_slots.py ... --slot_threshold 0.05 --save_json eval_t005.json

# Stricter (higher precision)
python evals/eval_point_slots.py ... --slot_threshold 0.08 --save_json eval_t008.json

# More tolerant matching
python evals/eval_point_slots.py ... --slot_threshold 0.05 --dist_thresh 12.0 --save_json eval_d12.json
```

---

## Optional: Debug & Inspection

### Visualize Ground Truth Projections

Verify GT points and line projections are correct:

```bash
python evals/debug_gt_projection.py \
  --manifest out_lineex_v3/val_manifest.jsonl \
  --out_dir outputs/debug_gt_projection \
  --start_index 0 \
  --num_samples 50 \
  --image_key full \
  --line_id 0 \
  --x_min 0 --x_max 10 \
  --y_min 0 --y_max 100
```

Useful for detecting:
- Incorrect GT positions
- Projection errors in older datasets
- Chart geometry issues

---

## Cluster Execution (Slurm)

### Train via Slurm

```bash
sbatch slurm_train_lineheatmap_slots.sh
```

### Predict via Slurm

```bash
sbatch slurm_predict_lineheatmap_slots.slurm
```

---

## Typical Development Cycle

```
1. Adjust config
   ↓
2. Train model
   ↓
3. Predict on validation
   ↓
4. Evaluate slot predictions
   ↓
5. Inspect visualizations
   ↓
6. Tune threshold / loss
   ↓
7. Retrain
```

**One-liner workflow:**

```bash
# Train
python -u -m app.lineheatmap2.train --config configs/lineheatmap/lineheatmap_vitl16v2.yaml

# Predict
python -u -m app.lineheatmap2.predict_heatmap \
  --config configs/lineheatmap/lineheatmap_vitl16v2.yaml \
  --checkpoint outputs/lineheatmap_vitl16_slots/best.pth \
  --num_samples 200 --start_index 0 --out_dir outputs/predictions_slots_v3

# Evaluate
python evals/eval_point_slots.py \
  --manifest out_lineex_v3/val_manifest.jsonl \
  --pred_dir outputs/predictions_slots_v3 \
  --x_min 0 --x_max 10 --y_min 0 --y_max 100 \
  --slot_threshold 0.05 --dist_thresh 10.0 --line_id 0 \
  --save_json outputs/predictions_slots_v3/eval_point_slots.json
```
---

## File Reference

**Generation & Data**
- `gen_linecharts3.py` — Synthetic chart generator
- `app/lineheatmap2/dataset.py` — Dataset builder

**Model & Training**
- `app/lineheatmap2/model.py` — Architecture
- `app/lineheatmap2/train.py` — Training loop

**Prediction & Evaluation**
- `app/lineheatmap2/predict_heatmap.py` — Inference
- `evals/eval_point_slots.py` — Evaluation
- `evals/debug_gt_projection.py` — GT visualization


