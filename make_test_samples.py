"""
make_test_samples.py
--------------------
Reproduces the EXACT same leakage-controlled train/validation/test split used to
train the model (group-aware, seed 42), then copies one image PER CLASS from the
held-out TEST set into a samples/ folder. This guarantees that the sample leaves
shown in the app are images the model never trained on, so clicking one is a
genuine live test on unseen data, not a replay of a memorised training image.

Run this on the machine that has the tomato dataset (the same one used to train).

Usage:
    python make_test_samples.py /path/to/tomato/train   ./samples

The first argument is the dataset folder that contains the 11 class subfolders
(the same folder the notebook loaded from). The second is where to write samples.
"""
import os
import re
import sys
import shutil
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ---- must match the notebook exactly ----
SEED = 42
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
CLASS_NAMES = [
    "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
    "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite", "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy", "powdery_mildew",
]


def base_image_id(filepath, label=""):
    """Identical to the notebook: class-scoped source-leaf id that collapses only
    explicit augmentation markers and rotation tokens."""
    fn = filepath.replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.(jpg|jpeg|png)$", "", fn, flags=re.I)
    s = re.sub(r"[_\-\s]*(flipped|flip|rotated|rotate|augmented|aug|copy)[_\-\s]*\d*", "", s, flags=re.I)
    s = re.sub(r"[_\-\s](90|180|270|360)$", "", s)
    s = re.sub(r"[_\-\s]+$", "", s)
    return f"{label}/{s.strip().lower()}"


def list_files_by_class(class_root, class_names):
    files, labels = [], []
    for i, c in enumerate(class_names):
        cdir = os.path.join(class_root, c)
        if not os.path.isdir(cdir):
            print(f"  warning: class folder not found: {cdir}")
            continue
        cfiles = [os.path.join(cdir, f) for f in os.listdir(cdir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        cfiles.sort()  # deterministic ordering so the split is reproducible
        files.extend(cfiles)
        labels.extend([i] * len(cfiles))
    return np.array(files), np.array(labels)


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_test_samples.py /path/to/tomato/train [./samples]")
        sys.exit(1)
    class_root = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./samples"

    files, labels = list_files_by_class(class_root, CLASS_NAMES)
    print(f"Loaded {len(files)} images across {len(set(labels))} classes.")

    # reproduce the identical group-aware split
    groups = np.array([base_image_id(p, CLASS_NAMES[l]) for p, l in zip(files, labels)])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    rest_idx, test_idx = next(gss1.split(files, labels, groups))

    f_test = files[test_idx]
    y_test = labels[test_idx]
    print(f"Reproduced test split: {len(f_test)} held-out images.")

    # verify zero leakage between the rest and the test set (sanity check)
    g_rest = set(groups[rest_idx])
    g_test = set(groups[test_idx])
    assert len(g_rest & g_test) == 0, "leakage detected: a source leaf is in both splits"
    print("Leakage check passed: no source leaf shared across splits.")

    # pick one TEST image per class
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    rng = np.random.default_rng(SEED)
    picked = 0
    for i, c in enumerate(CLASS_NAMES):
        idx = np.where(y_test == i)[0]
        if len(idx) == 0:
            print(f"  warning: no test image for class {c}")
            continue
        chosen = f_test[rng.choice(idx)]
        ext = os.path.splitext(chosen)[1].lower()
        dest = os.path.join(out_dir, f"{c}{ext}")
        shutil.copy(chosen, dest)
        picked += 1
        print(f"  {c:<38} <- {os.path.basename(chosen)}")

    print(f"\nDone. {picked} held-out test images written to {out_dir}/")
    print("These are genuinely unseen by the model, so the app demo is a real test.")


if __name__ == "__main__":
    main()
