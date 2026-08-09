"""
=============================================================================
  YOLOv11  Training Pipeline  —  Optimised for Small Datasets
=============================================================================
  Dataset   : C:\\Users\\meljo\\krab-3\\dataset_yolo   (170 imgs, 6 classes)
  Hardware  : NVIDIA RTX 5050 (laptop)  |  Python 3.x  |  CUDA
  Model     : YOLOv11s  (best accuracy/speed for ~170 images)
  Target    : Maximum mAP with heavy augmentation + transfer learning

  Run       : python train_yolo11.py
=============================================================================
"""

import os
import sys
import shutil
import logging
import time
import math
from pathlib import Path
from datetime import datetime

# ── Verify ultralytics is installed ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("[ERROR] ultralytics or torch not installed.")
    print("        Run:  pip install ultralytics")
    sys.exit(1)

# =============================================================================
#  ██████████  CONFIGURATION  ██████████
#  ← Only section you ever need to touch
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_YAML   = r"C:\Users\meljo\krab-3\dataset_yolo\data.yaml"
PROJECT_DIR    = r"C:\Users\meljo\krab-3\runs"      # all run folders go here
RUN_NAME       = "obstacle_v1"                       # subfolder for this run

# ── Model ─────────────────────────────────────────────────────────────────────
# yolo11n < yolo11s < yolo11m < yolo11l < yolo11x  (size vs accuracy)
# For 170 images: yolo11s  →  best trade-off.  yolo11m if you want to push further.
MODEL_VARIANT  = "yolo11s.pt"   # downloads pretrained COCO weights automatically

# ── Core training params ──────────────────────────────────────────────────────
EPOCHS         = 150
IMG_SIZE       = 640            # standard; RTX 5050 handles this fine at batch 16
BATCH_SIZE     = 16             # RTX 5050 ~8GB VRAM  →  16 is safe at imgsz 640
                                 # drop to 8 if you hit CUDA OOM
WORKERS        = 4              # data-loader threads (Windows: keep ≤ 4)

# ── Optimiser & LR ────────────────────────────────────────────────────────────
OPTIMIZER      = "AdamW"        # AdamW > SGD for small datasets
LR0            = 0.001          # initial LR  (lower = safer for fine-tuning)
LRF            = 0.01           # final LR = LR0 * LRF  (cosine decay target)
MOMENTUM       = 0.937
WEIGHT_DECAY   = 0.0005

# ── Warmup ────────────────────────────────────────────────────────────────────
WARMUP_EPOCHS  = 5              # ramp up LR for first N epochs
WARMUP_BIAS_LR = 0.1

# ── Regularisation ────────────────────────────────────────────────────────────
DROPOUT        = 0.0            # YOLOv11 backbone dropout (0 = off for small models)
LABEL_SMOOTHING= 0.05           # prevents overconfidence  (key for small datasets)

# ── Early stopping ────────────────────────────────────────────────────────────
PATIENCE       = 30             # stop if no mAP improvement for 30 epochs

# ── Transfer learning  ────────────────────────────────────────────────────────
# Freeze the backbone for the first N epochs so the pretrained features
# don't get destroyed by your small dataset.  Unfreeze after warmup.
FREEZE_LAYERS  = 10             # freeze first 10 layers (backbone)

# ── Augmentation  ─────────────────────────────────────────────────────────────
# Heavy augmentation is the main weapon against overfitting on small datasets.
MOSAIC         = 1.0            # mosaic prob (4-image collage)  — keep at 1.0
MIXUP          = 0.15           # blends two images  — helps generalisation
COPY_PASTE     = 0.1            # paste objects across images
CLOSE_MOSAIC   = 15             # disable mosaic for last N epochs (stabilise)

# Geometric
DEGREES        = 10.0           # rotation ± degrees
TRANSLATE      = 0.15           # translation fraction
SCALE          = 0.6            # scale  ± fraction  (important for object sizes)
SHEAR          = 2.0            # shear  ± degrees
PERSPECTIVE    = 0.0001         # perspective distortion

# Flip
FLIPLR         = 0.5            # horizontal flip prob
FLIPUD         = 0.05           # vertical flip prob  (rarely useful, low)

# Color / photometric
HSV_H          = 0.015          # hue shift
HSV_S          = 0.7            # saturation shift  (simulate lighting)
HSV_V          = 0.4            # brightness shift

# ── Loss weights ─────────────────────────────────────────────────────────────
# Tuned for multi-class obstacle/object detection
BOX_LOSS_GAIN  = 7.5
CLS_LOSS_GAIN  = 0.5            # lower class weight → less overfit on rare classes
DFL_LOSS_GAIN  = 1.5

# ── Misc ──────────────────────────────────────────────────────────────────────
CONF_THRESH    = 0.25           # confidence threshold for validation
IOU_THRESH     = 0.7            # NMS IoU threshold
SAVE_PERIOD    = 10             # save checkpoint every N epochs (0 = only best/last)
HALF_PRECISION = True           # FP16 mixed precision  →  faster on RTX 5050
VERBOSE        = True


# =============================================================================
#  ██████████  LOGGING  ██████████
# =============================================================================

os.makedirs(PROJECT_DIR, exist_ok=True)
_log_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = os.path.join(PROJECT_DIR, f"train_{_log_ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
#  ██████████  ENVIRONMENT CHECK  ██████████
# =============================================================================

def check_environment():
    log.info("=" * 62)
    log.info("  ENVIRONMENT CHECK")
    log.info("=" * 62)

    # Python
    log.info(f"  Python      : {sys.version.split()[0]}")

    # PyTorch + CUDA
    log.info(f"  PyTorch     : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    log.info(f"  CUDA        : {'YES  ✓' if cuda_ok else 'NO  ✗  (will use CPU — slow!)'}")
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  GPU         : {gpu_name}")
        log.info(f"  VRAM        : {vram_gb:.1f} GB")

    # Ultralytics
    import ultralytics
    log.info(f"  Ultralytics : {ultralytics.__version__}")

    # Dataset YAML
    if not Path(DATASET_YAML).exists():
        log.error(f"  data.yaml NOT found: {DATASET_YAML}")
        sys.exit(1)
    log.info(f"  data.yaml   : OK  ({DATASET_YAML})")

    # Count images
    ds_root = Path(DATASET_YAML).parent
    n_train = len(list((ds_root / "images" / "train").glob("*")))
    n_val   = len(list((ds_root / "images" / "val").glob("*")))
    log.info(f"  Train imgs  : {n_train}")
    log.info(f"  Val imgs    : {n_val}")
    log.info("=" * 62)

    if not cuda_ok:
        log.warning("CUDA not available — training will be very slow on CPU.")
        ans = input("Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit(0)

    return cuda_ok


# =============================================================================
#  ██████████  DATA YAML PATCH  ██████████
#  Ultralytics needs the path: field to point to an absolute location.
#  We rewrite it programmatically so Windows paths work correctly.
# =============================================================================

def patch_data_yaml(yaml_path: str) -> str:
    """
    Ensure data.yaml has an absolute path: field.
    Returns the (possibly rewritten) yaml path.
    """
    yaml_path = Path(yaml_path)
    text = yaml_path.read_text(encoding="utf-8")

    # Replace or insert path: with absolute posix path
    ds_root = str(yaml_path.parent).replace("\\", "/")
    lines   = []
    patched = False
    for line in text.splitlines():
        if line.strip().startswith("path:"):
            lines.append(f"path: {ds_root}")
            patched = True
        else:
            lines.append(line)
    if not patched:
        lines.insert(0, f"path: {ds_root}")

    new_text = "\n".join(lines)
    yaml_path.write_text(new_text, encoding="utf-8")
    log.info(f"data.yaml patched  →  path: {ds_root}")
    return str(yaml_path)


# =============================================================================
#  ██████████  TRAINING  ██████████
# =============================================================================

def train(cuda_ok: bool):
    log.info("Loading model …")
    model = YOLO(MODEL_VARIANT)   # downloads pretrained weights if not cached

    log.info(f"Starting training — {EPOCHS} epochs  |  {MODEL_VARIANT}")
    log.info(f"Project : {PROJECT_DIR}  /  {RUN_NAME}")
    log.info("-" * 62)

    t0 = time.time()

    results = model.train(
        # ── Data ──────────────────────────────────────────────────────────
        data        = DATASET_YAML,
        imgsz       = IMG_SIZE,
        batch       = BATCH_SIZE,
        workers     = WORKERS,

        # ── Training schedule ─────────────────────────────────────────────
        epochs      = EPOCHS,
        patience    = PATIENCE,
        close_mosaic= CLOSE_MOSAIC,

        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer   = OPTIMIZER,
        lr0         = LR0,
        lrf         = LRF,
        momentum    = MOMENTUM,
        weight_decay= WEIGHT_DECAY,
        warmup_epochs    = WARMUP_EPOCHS,
        warmup_bias_lr   = WARMUP_BIAS_LR,

        # ── Regularisation ────────────────────────────────────────────────
        label_smoothing  = LABEL_SMOOTHING,
        dropout          = DROPOUT,

        # ── Transfer learning ─────────────────────────────────────────────
        freeze      = FREEZE_LAYERS,

        # ── Loss weights ──────────────────────────────────────────────────
        box         = BOX_LOSS_GAIN,
        cls         = CLS_LOSS_GAIN,
        dfl         = DFL_LOSS_GAIN,

        # ── Augmentation ──────────────────────────────────────────────────
        mosaic      = MOSAIC,
        mixup       = MIXUP,
        copy_paste  = COPY_PASTE,
        degrees     = DEGREES,
        translate   = TRANSLATE,
        scale       = SCALE,
        shear       = SHEAR,
        perspective = PERSPECTIVE,
        fliplr      = FLIPLR,
        flipud      = FLIPUD,
        hsv_h       = HSV_H,
        hsv_s       = HSV_S,
        hsv_v       = HSV_V,

        # ── Evaluation ────────────────────────────────────────────────────
        conf        = CONF_THRESH,
        iou         = IOU_THRESH,

        # ── Hardware ──────────────────────────────────────────────────────
        device      = 0 if cuda_ok else "cpu",
        half        = HALF_PRECISION and cuda_ok,

        # ── Output ────────────────────────────────────────────────────────
        project     = PROJECT_DIR,
        name        = RUN_NAME,
        save_period = SAVE_PERIOD,
        plots       = True,       # saves training curves, confusion matrix, PR curve
        verbose     = VERBOSE,
        exist_ok    = True,
    )

    elapsed = time.time() - t0
    return results, elapsed


# =============================================================================
#  ██████████  POST-TRAINING VALIDATION  ██████████
# =============================================================================

def validate_best(cuda_ok: bool):
    """Run val on the best saved weights and print per-class metrics."""
    best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        log.warning("best.pt not found — skipping standalone validation.")
        return

    log.info("=" * 62)
    log.info("  RUNNING VALIDATION  on best.pt")
    log.info("=" * 62)

    model = YOLO(str(best_weights))
    metrics = model.val(
        data    = DATASET_YAML,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        device  = 0 if cuda_ok else "cpu",
        half    = HALF_PRECISION and cuda_ok,
        plots   = True,
        verbose = True,
    )
    return metrics


# =============================================================================
#  ██████████  SUMMARY REPORT  ██████████
# =============================================================================

def print_final_report(results, elapsed: float):
    run_dir    = Path(PROJECT_DIR) / RUN_NAME
    best_wts   = run_dir / "weights" / "best.pt"
    last_wts   = run_dir / "weights" / "last.pt"

    # Pull best metrics from results object
    try:
        map50    = results.results_dict.get("metrics/mAP50(B)",    0)
        map5095  = results.results_dict.get("metrics/mAP50-95(B)", 0)
        prec     = results.results_dict.get("metrics/precision(B)", 0)
        rec      = results.results_dict.get("metrics/recall(B)",    0)
    except Exception:
        map50 = map5095 = prec = rec = 0.0

    h  = int(elapsed // 3600)
    m  = int((elapsed % 3600) // 60)
    s  = int(elapsed % 60)

    log.info("=" * 62)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 62)
    log.info(f"  Total time         : {h:02d}h {m:02d}m {s:02d}s")
    log.info(f"  mAP@0.5            : {map50:.4f}")
    log.info(f"  mAP@0.5:0.95       : {map5095:.4f}")
    log.info(f"  Precision          : {prec:.4f}")
    log.info(f"  Recall             : {rec:.4f}")
    log.info("=" * 62)
    log.info(f"  Best weights       : {best_wts}")
    log.info(f"  Last weights       : {last_wts}")
    log.info(f"  Training plots     : {run_dir}")
    log.info(f"  Session log        : {_log_file}")
    log.info("=" * 62)

    log.info("\n  ARTIFACTS SAVED IN RUN FOLDER:")
    artifacts = {
        "confusion_matrix.png"        : "Class prediction accuracy grid",
        "PR_curve.png"                : "Precision-Recall curve per class",
        "F1_curve.png"                : "F1 score vs confidence threshold",
        "results.png"                 : "All training curves in one image",
        "results.csv"                 : "Epoch-by-epoch metrics (open in Excel)",
        "val_batch0_pred.jpg"         : "Sample val-batch predictions",
        "weights/best.pt"             : "Best checkpoint  ← USE THIS",
        "weights/last.pt"             : "Last epoch checkpoint",
    }
    for fname, desc in artifacts.items():
        log.info(f"    {fname:<38}  {desc}")

    log.info("=" * 62)
    log.info("  HOW TO USE best.pt:")
    log.info("")
    log.info("    from ultralytics import YOLO")
    log.info(f"    model = YOLO(r'{best_wts}')")
    log.info("    results = model.predict('your_image.jpg', conf=0.25)")
    log.info("=" * 62)


# =============================================================================
#  ██████████  MAIN  ██████████
# =============================================================================

def main():
    log.info("YOLOv11 Training Pipeline  —  Obstacle Detection")
    log.info(f"Run timestamp : {_log_ts}")

    # 1. Environment check
    cuda_ok = check_environment()

    # 2. Patch data.yaml with absolute path
    patch_data_yaml(DATASET_YAML)

    # 3. Train
    results, elapsed = train(cuda_ok)

    # 4. Standalone validation on best weights
    validate_best(cuda_ok)

    # 5. Final report
    print_final_report(results, elapsed)


if __name__ == "__main__":
    main()