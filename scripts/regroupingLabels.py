from pathlib import Path
from collections import Counter
from shutil import copy2
import datetime as dt

# =========================
# CONFIG
# =========================
DATASET_ROOT = Path(r"C:\Users\meljo\krab-3\dataset_yolo")

LABELS_DIRS = [
    DATASET_ROOT / "labels" / "train",
    DATASET_ROOT / "labels" / "val",
]

DATA_YAML = DATASET_ROOT / "data.yaml"

# New grouped class order
NEW_CLASSES = [
    "box",          # box1, box2, box3, bluebox
    "sticky_note",  # note_sticky, stickynote
    "bottle",       # bottle
    "plastic",      # plastic_cover
    "table",        # table
    "tape",         # tape
]

# Old YOLO class ids from your conversion log:
# [0] bluebox
# [1] bottle
# [2] box1
# [3] box2
# [4] box3
# [5] note_sticky
# [6] plastic_cover
# [7] stickynote
# [8] table
# [9] tape
OLD_TO_NEW = {
    0: 0,  # bluebox -> box
    1: 2,  # bottle -> bottle
    2: 0,  # box1 -> box
    3: 0,  # box2 -> box
    4: 0,  # box3 -> box
    5: 1,  # note_sticky -> sticky_note
    6: 3,  # plastic_cover -> plastic
    7: 1,  # stickynote -> sticky_note
    8: 4,  # table -> table
    9: 5,  # tape -> tape
}

# =========================
# HELPERS
# =========================
def backup_file(src: Path) -> None:
    """Create a .bak copy next to the original file."""
    bak = src.with_suffix(src.suffix + ".bak")
    if not bak.exists():
        copy2(src, bak)

def remap_label_file(label_path: Path) -> tuple[int, Counter]:
    """
    Remap class ids inside one YOLO label file.
    Returns: (num_lines_written, class_counter)
    """
    if not label_path.exists():
        return 0, Counter()

    backup_file(label_path)

    new_lines = []
    class_counter = Counter()

    with label_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                # malformed line, skip it
                continue

            try:
                old_class_id = int(parts[0])
            except ValueError:
                continue

            if old_class_id not in OLD_TO_NEW:
                continue

            new_class_id = OLD_TO_NEW[old_class_id]
            new_line = " ".join([str(new_class_id)] + parts[1:5])
            new_lines.append(new_line)
            class_counter[new_class_id] += 1

    with label_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    return len(new_lines), class_counter

def rewrite_data_yaml() -> None:
    """
    Overwrite data.yaml with the grouped classes.
    Keeps your split paths the same.
    """
    content = f"""# Auto-generated grouped YOLO dataset file
path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val

nc: {len(NEW_CLASSES)}
names:
"""
    for i, name in enumerate(NEW_CLASSES):
        content += f"  {i}: {name}\n"

    # backup existing yaml
    if DATA_YAML.exists():
        backup_file(DATA_YAML)

    DATA_YAML.write_text(content, encoding="utf-8")

def main() -> None:
    print("=== Grouping YOLO classes in existing split ===")
    print(f"Dataset root : {DATASET_ROOT}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    total_files = 0
    total_lines = 0
    overall_counter = Counter()

    for folder in LABELS_DIRS:
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue

        for label_file in folder.glob("*.txt"):
            total_files += 1
            lines_written, counter = remap_label_file(label_file)
            total_lines += lines_written
            overall_counter.update(counter)

    rewrite_data_yaml()

    print("\n=== Done ===")
    print(f"Label files processed : {total_files}")
    print(f"Total label lines kept : {total_lines}")
    print(f"Updated data.yaml      : {DATA_YAML}")

    print("\nNew grouped class order:")
    for i, name in enumerate(NEW_CLASSES):
        print(f"  [{i}] {name}")

    print("\nClass instance counts after grouping:")
    for i, name in enumerate(NEW_CLASSES):
        print(f"  {name:12s}: {overall_counter[i]}")

    print("\nBackups created:")
    print("  - Each original .txt file saved as .txt.bak")
    if DATA_YAML.exists():
        print("  - data.yaml saved as data.yaml.bak")

if __name__ == "__main__":
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Run time: {timestamp}")
    main()