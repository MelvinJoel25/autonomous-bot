"""
=============================================================================
Pascal VOC XML  →  YOLO TXT  Conversion Pipeline
=============================================================================
Author  : Production-ready script for Melvin Joel
Dataset : C:\\Users\\meljo\\krab-3\\new dataset
Usage   : python voc_to_yolo.py
=============================================================================
"""

import os
import sys
import xml.etree.ElementTree as ET
import shutil
import random
import logging
import math
from pathlib import Path

# ── Try importing cv2 (optional – only used for preview step) ──────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python not found. Preview step will be skipped.")
    print("         Install with: pip install opencv-python\n")

# =============================================================================
# ██████████████  CONFIGURATION  ██████████████
# =============================================================================

# ── Root of your dataset ──────────────────────────────────────────────────
DATASET_ROOT = r"C:\Users\meljo\krab-3\new dataset"

# ── Sub-folder names inside DATASET_ROOT (change if yours differ) ─────────
IMAGES_SUBFOLDER = "images"     # folder containing .jpg / .png images
LABELS_SUBFOLDER = "label"     # folder containing .xml annotation files
                                 # (can also be "annotations", "Annotations", etc.)

# ── Output directory (will be created automatically) ─────────────────────
OUTPUT_DIR = r"C:\Users\meljo\krab-3\dataset_yolo"

# ── Train / Val split ratio ───────────────────────────────────────────────
TRAIN_RATIO = 0.80   # 80% train, 20% val

# ── Number of preview images to generate ─────────────────────────────────
NUM_PREVIEWS = 5

# ── Accepted image extensions ────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── OPTIONAL: Hard-code your class list here if you want a fixed order.
#    Leave as None to auto-detect from XML files (recommended).
FIXED_CLASSES = None   # e.g. ["robot", "obstacle", "bot_front"]

# =============================================================================
# ██████████████  LOGGING SETUP  ██████████████
# =============================================================================

LOG_FILE = os.path.join(OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else ".", "conversion.log")

def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        ],
    )

log = logging.getLogger(__name__)


# =============================================================================
# ██████████████  STEP 1 – DISCOVER DATASET PATHS  ██████████████
# =============================================================================

def discover_paths(root: str, img_sub: str, lbl_sub: str):
    """
    Finds the images folder and labels folder inside root.
    Falls back to searching common alternative names if the defaults
    don't exist.
    """
    root = Path(root)
    if not root.exists():
        log.error(f"DATASET_ROOT does not exist: {root}")
        sys.exit(1)

    # ── Search for images folder ──────────────────────────────────────────
    img_candidates = [img_sub, "Images", "JPEGImages", "imgs", "image"]
    img_dir = None
    for cand in img_candidates:
        p = root / cand
        if p.is_dir():
            img_dir = p
            break
    if img_dir is None:
        # Last resort: any folder that contains image files
        for p in root.iterdir():
            if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.iterdir()):
                img_dir = p
                break
    if img_dir is None:
        log.error("Cannot find an images folder inside the dataset root.")
        sys.exit(1)

    # ── Search for labels/annotations folder ─────────────────────────────
    lbl_candidates = [lbl_sub, "Labels", "annotations", "Annotations", "xmls"]
    lbl_dir = None
    for cand in lbl_candidates:
        p = root / cand
        if p.is_dir():
            lbl_dir = p
            break
    if lbl_dir is None:
        # Maybe XML files sit alongside images
        xml_in_img = list(img_dir.glob("*.xml"))
        if xml_in_img:
            lbl_dir = img_dir
    if lbl_dir is None:
        log.error("Cannot find a label/annotations folder inside the dataset root.")
        sys.exit(1)

    log.info(f"Images folder  : {img_dir}")
    log.info(f"Labels folder  : {lbl_dir}")
    return img_dir, lbl_dir


# =============================================================================
# ██████████████  STEP 2 – AUTO-DETECT CLASSES  ██████████████
# =============================================================================

def collect_classes(lbl_dir: Path) -> list:
    """Scan all XML files and collect unique class names in sorted order."""
    classes = set()
    for xml_file in lbl_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is not None and name_el.text:
                    classes.add(name_el.text.strip())
        except ET.ParseError:
            pass   # corrupt files handled later
    return sorted(classes)


# =============================================================================
# ██████████████  STEP 3 – PARSE ONE XML FILE  ██████████████
# =============================================================================

def parse_voc_xml(xml_path: Path, class_map: dict):
    """
    Parse a Pascal VOC XML annotation file.

    Returns
    -------
    (img_w, img_h, yolo_lines, skipped_objects)
        img_w, img_h  : integers from <size> tag
        yolo_lines    : list of "class_id cx cy w h" strings (normalised 0-1)
        skipped_objs  : list of dicts describing every skipped bounding box
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Corrupt XML: {e}")

    root = tree.getroot()

    # ── Read image size ───────────────────────────────────────────────────
    size_el = root.find("size")
    if size_el is None:
        raise ValueError("Missing <size> tag")

    def _int(tag):
        el = size_el.find(tag)
        if el is None or not el.text:
            raise ValueError(f"Missing <{tag}> in <size>")
        return int(el.text.strip())

    img_w = _int("width")
    img_h = _int("height")

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {img_w}×{img_h}")

    yolo_lines  = []
    skipped_objs = []

    for obj in root.findall("object"):
        # ── Class name ────────────────────────────────────────────────────
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            skipped_objs.append({"reason": "missing <name>"})
            continue
        class_name = name_el.text.strip()
        if class_name not in class_map:
            skipped_objs.append({"reason": f"unknown class '{class_name}'"})
            continue
        class_id = class_map[class_name]

        # ── Bounding box ──────────────────────────────────────────────────
        bndbox = obj.find("bndbox")
        if bndbox is None:
            skipped_objs.append({"class": class_name, "reason": "missing <bndbox>"})
            continue

        try:
            xmin = float(bndbox.find("xmin").text.strip())
            ymin = float(bndbox.find("ymin").text.strip())
            xmax = float(bndbox.find("xmax").text.strip())
            ymax = float(bndbox.find("ymax").text.strip())
        except (AttributeError, ValueError, TypeError) as e:
            skipped_objs.append({"class": class_name, "reason": f"bad bbox coords: {e}"})
            continue

        # ── Clamp to image bounds ─────────────────────────────────────────
        xmin_c = max(0.0, min(xmin, img_w))
        ymin_c = max(0.0, min(ymin, img_h))
        xmax_c = max(0.0, min(xmax, img_w))
        ymax_c = max(0.0, min(ymax, img_h))

        if xmin_c >= xmax_c or ymin_c >= ymax_c:
            skipped_objs.append({
                "class": class_name,
                "reason": f"degenerate bbox after clamping "
                          f"[{xmin},{ymin},{xmax},{ymax}] → "
                          f"[{xmin_c},{ymin_c},{xmax_c},{ymax_c}]",
            })
            continue

        if (xmin != xmin_c or ymin != ymin_c or
                xmax != xmax_c or ymax != ymax_c):
            skipped_objs.append({
                "class": class_name,
                "reason": f"bbox clamped from [{xmin},{ymin},{xmax},{ymax}] "
                          f"to [{xmin_c},{ymin_c},{xmax_c},{ymax_c}]",
                "severity": "warning_only",
            })

        # ── Convert to YOLO normalised format ─────────────────────────────
        cx = ((xmin_c + xmax_c) / 2.0) / img_w
        cy = ((ymin_c + ymax_c) / 2.0) / img_h
        bw = (xmax_c - xmin_c) / img_w
        bh = (ymax_c - ymin_c) / img_h

        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return img_w, img_h, yolo_lines, skipped_objs


# =============================================================================
# ██████████████  STEP 4 – BUILD OUTPUT FOLDER STRUCTURE  ██████████████
# =============================================================================

def create_output_dirs(out: str) -> dict:
    """Create YOLO-compatible directory tree and return path dict."""
    dirs = {
        "root"       : Path(out),
        "img_train"  : Path(out) / "images"  / "train",
        "img_val"    : Path(out) / "images"  / "val",
        "lbl_train"  : Path(out) / "labels"  / "train",
        "lbl_val"    : Path(out) / "labels"  / "val",
        "previews"   : Path(out) / "previews",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# =============================================================================
# ██████████████  STEP 5 – MAIN CONVERSION LOOP  ██████████████
# =============================================================================

def convert_dataset(img_dir: Path, lbl_dir: Path, class_map: dict,
                    dirs: dict, train_ratio: float):
    """
    Match every image to its XML, convert, and copy into the YOLO structure.

    Returns a summary dict.
    """
    # Collect all image files
    all_images = sorted(
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not all_images:
        log.error("No image files found in the images folder.")
        sys.exit(1)

    log.info(f"Total images found: {len(all_images)}")

    # Shuffle then split
    random.seed(42)
    random.shuffle(all_images)
    split_idx   = math.ceil(len(all_images) * train_ratio)
    train_imgs  = all_images[:split_idx]
    val_imgs    = all_images[split_idx:]
    log.info(f"Split → train: {len(train_imgs)}  |  val: {len(val_imgs)}")

    summary = {
        "total"          : len(all_images),
        "converted"      : 0,
        "skipped_no_xml" : 0,
        "skipped_corrupt": 0,
        "skipped_no_objs": 0,
        "warnings"       : 0,
        "skipped_detail" : [],
    }

    converted_pairs = []   # (img_path, split) for preview step

    def process_split(img_list, split_name):
        img_out = dirs[f"img_{split_name}"]
        lbl_out = dirs[f"lbl_{split_name}"]

        for img_path in img_list:
            stem    = img_path.stem
            xml_path = lbl_dir / (stem + ".xml")

            # ── Missing XML ───────────────────────────────────────────────
            if not xml_path.exists():
                log.warning(f"[SKIP – no XML]  {img_path.name}")
                summary["skipped_no_xml"] += 1
                summary["skipped_detail"].append(
                    {"file": img_path.name, "reason": "no matching XML"})
                continue

            # ── Parse XML ─────────────────────────────────────────────────
            try:
                _, _, yolo_lines, skipped_objs = parse_voc_xml(xml_path, class_map)
            except ValueError as e:
                log.warning(f"[SKIP – corrupt]  {img_path.name}  →  {e}")
                summary["skipped_corrupt"] += 1
                summary["skipped_detail"].append(
                    {"file": img_path.name, "reason": str(e)})
                continue

            # Log per-object warnings
            for sk in skipped_objs:
                sev = sk.get("severity", "error")
                if sev == "warning_only":
                    log.warning(f"  [BBOX CLAMPED]  {img_path.name}  →  {sk['reason']}")
                    summary["warnings"] += 1
                else:
                    log.warning(f"  [OBJ SKIPPED]   {img_path.name}  →  {sk['reason']}")

            if not yolo_lines:
                log.warning(f"[SKIP – no objs]  {img_path.name}")
                summary["skipped_no_objs"] += 1
                summary["skipped_detail"].append(
                    {"file": img_path.name, "reason": "zero valid objects after conversion"})
                continue

            # ── Write YOLO label file ─────────────────────────────────────
            lbl_file = lbl_out / (stem + ".txt")
            lbl_file.write_text("\n".join(yolo_lines), encoding="utf-8")

            # ── Copy image ────────────────────────────────────────────────
            shutil.copy2(img_path, img_out / img_path.name)

            summary["converted"] += 1
            converted_pairs.append((img_path, split_name))

    process_split(train_imgs, "train")
    process_split(val_imgs,   "val")

    return summary, converted_pairs


# =============================================================================
# ██████████████  STEP 6 – GENERATE data.yaml  ██████████████
# =============================================================================

def write_data_yaml(out_root: Path, class_names: list):
    """Write YOLOv8/v11-compatible data.yaml."""
    yaml_path = out_root / "data.yaml"
    lines = [
        f"path: {out_root.as_posix()}   # dataset root",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(class_names)}   # number of classes",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"data.yaml written → {yaml_path}")
    return yaml_path


# =============================================================================
# ██████████████  STEP 7 – PREVIEW / VISUALISATION  ██████████████
# =============================================================================

# Colour palette (BGR) – one per class
PALETTE = [
    (0,   255,  0),    # green
    (255,  0,   0),    # blue
    (0,   0,   255),   # red
    (0,   255, 255),   # yellow
    (255, 0,   255),   # magenta
    (255, 165,  0),    # orange
    (128, 0,   128),   # purple
    (0,   128, 128),   # teal
]

def draw_yolo_boxes(img_bgr, lbl_path: Path, class_names: list):
    """Draw YOLO bounding boxes on an image (modifies in-place)."""
    h, w = img_bgr.shape[:2]
    if not lbl_path.exists():
        return img_bgr
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid, cx, cy, bw, bh = int(parts[0]), float(parts[1]), \
                               float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = PALETTE[cid % len(PALETTE)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = class_names[cid] if cid < len(class_names) else str(cid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img_bgr


def generate_previews(converted_pairs: list, dirs: dict,
                      class_names: list, n: int = 5):
    """Pick n random converted images and save annotated previews."""
    if not CV2_AVAILABLE:
        log.warning("Skipping preview generation (cv2 not available).")
        return

    if not converted_pairs:
        log.warning("No converted pairs available for preview.")
        return

    picks = random.sample(converted_pairs, min(n, len(converted_pairs)))
    saved = 0
    for img_path, split in picks:
        stem = img_path.stem
        # Find copied image in output folder
        copied_img = dirs[f"img_{split}"] / img_path.name
        lbl_file   = dirs[f"lbl_{split}"] / (stem + ".txt")

        img = cv2.imread(str(copied_img))
        if img is None:
            log.warning(f"Preview: could not read {copied_img}")
            continue

        img = draw_yolo_boxes(img, lbl_file, class_names)
        out_path = dirs["previews"] / f"preview_{split}_{img_path.name}"
        cv2.imwrite(str(out_path), img)
        log.info(f"Preview saved → {out_path.name}")
        saved += 1

    log.info(f"Total previews saved: {saved}")


# =============================================================================
# ██████████████  STEP 8 – SUMMARY REPORT  ██████████████
# =============================================================================

def print_summary(summary: dict, class_names: list, dirs: dict):
    sep = "=" * 62
    log.info(sep)
    log.info("  CONVERSION SUMMARY")
    log.info(sep)
    log.info(f"  Total images found       : {summary['total']}")
    log.info(f"  Successfully converted   : {summary['converted']}")
    log.info(f"  Skipped (no XML found)   : {summary['skipped_no_xml']}")
    log.info(f"  Skipped (corrupt XML)    : {summary['skipped_corrupt']}")
    log.info(f"  Skipped (no valid objs)  : {summary['skipped_no_objs']}")
    log.info(f"  Bbox clamp warnings      : {summary['warnings']}")
    log.info(sep)
    log.info(f"  Classes ({len(class_names)}):")
    for i, name in enumerate(class_names):
        log.info(f"    [{i}]  {name}")
    log.info(sep)
    log.info(f"  Output folder  : {dirs['root']}")
    log.info(f"  Log file       : {LOG_FILE}")
    log.info(sep)

    if summary["skipped_detail"]:
        log.info("  SKIPPED FILES DETAIL:")
        for item in summary["skipped_detail"]:
            log.info(f"    {item['file']}  →  {item['reason']}")
        log.info(sep)


# =============================================================================
# ██████████████  MAIN  ██████████████
# =============================================================================

def main():
    # ── 0. Logging ───────────────────────────────────────────────────────
    # We need OUTPUT_DIR to exist first for the log file
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    setup_logging()
    log.info("Pascal VOC  →  YOLO Conversion Pipeline")
    log.info(f"Dataset root : {DATASET_ROOT}")
    log.info(f"Output dir   : {OUTPUT_DIR}")

    # ── 1. Discover paths ────────────────────────────────────────────────
    img_dir, lbl_dir = discover_paths(DATASET_ROOT, IMAGES_SUBFOLDER, LABELS_SUBFOLDER)

    # ── 2. Class detection ───────────────────────────────────────────────
    if FIXED_CLASSES:
        class_names = FIXED_CLASSES
        log.info(f"Using fixed class list: {class_names}")
    else:
        log.info("Auto-detecting classes from XML files …")
        class_names = collect_classes(lbl_dir)
        if not class_names:
            log.error("No classes found in XML files. Check your label folder.")
            sys.exit(1)
        log.info(f"Detected {len(class_names)} class(es): {class_names}")

    class_map = {name: idx for idx, name in enumerate(class_names)}

    # ── 3. Create output folder structure ────────────────────────────────
    dirs = create_output_dirs(OUTPUT_DIR)

    # ── 4. Convert ───────────────────────────────────────────────────────
    summary, converted_pairs = convert_dataset(
        img_dir, lbl_dir, class_map, dirs, TRAIN_RATIO
    )

    # ── 5. data.yaml ─────────────────────────────────────────────────────
    write_data_yaml(dirs["root"], class_names)

    # ── 6. Previews ──────────────────────────────────────────────────────
    log.info(f"Generating {NUM_PREVIEWS} preview images …")
    generate_previews(converted_pairs, dirs, class_names, NUM_PREVIEWS)

    # ── 7. Summary ───────────────────────────────────────────────────────
    print_summary(summary, class_names, dirs)


if __name__ == "__main__":
    main()