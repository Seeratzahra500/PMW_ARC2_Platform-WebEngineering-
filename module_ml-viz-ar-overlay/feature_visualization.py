#!/usr/bin/env python3
"""
ML Visualization — Feature Keypoint Detection
===============================================
Goal: Visualize the ML/CV features a Structure-from-Motion pipeline actually
uses to understand an image — the same SIFT keypoints COLMAP extracts before
it can match images together or reconstruct 3D structure.

This is a legitimate "ML visualization" deliverable on its own (it shows what
the model "sees" in a single photo) and it doubles as a diagnostic: a gate
photo with too few / poorly distributed keypoints will also struggle in the
capture-to-3D stage later.

Usage:
    python3 feature_visualization.py --input photo.jpg --output out.png
    python3 feature_visualization.py --input folder_of_photos/ --output-dir viz/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def visualize_keypoints(img_path: Path, out_path: Path, n_features: int = 0):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=n_features)  # 0 = unlimited
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Rich keypoints: circle size ~ scale, line ~ orientation
    vis = cv2.drawKeypoints(
        img, keypoints, None,
        color=(60, 200, 255),  # BGR — warm terracotta-ish highlight
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # Simple stats overlay
    n_kp = len(keypoints)
    responses = np.array([kp.response for kp in keypoints]) if n_kp else np.array([0])
    label = f"{n_kp} SIFT keypoints  |  avg strength {responses.mean():.4f}"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 620), 40), (30, 24, 20), -1)
    cv2.putText(vis, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (231, 220, 200), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return n_kp


def main():
    parser = argparse.ArgumentParser(description="SIFT keypoint ML visualization")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--output", help="Output image path (single-file mode)")
    parser.add_argument("--output-dir", help="Output folder (batch mode, for a folder input)")
    parser.add_argument("--n-features", type=int, default=0, help="Max keypoints (0 = unlimited)")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}

    if in_path.is_dir():
        out_dir = Path(args.output_dir or "viz_keypoints")
        images = sorted([p for p in in_path.iterdir() if p.suffix.lower() in exts])
        if not images:
            print(f"No images found in {in_path}")
            sys.exit(1)
        for img_path in images:
            out_path = out_dir / f"{img_path.stem}_keypoints.png"
            n_kp = visualize_keypoints(img_path, out_path, args.n_features)
            print(f"{img_path.name}: {n_kp} keypoints -> {out_path}")
    else:
        out_path = Path(args.output or f"{in_path.stem}_keypoints.png")
        n_kp = visualize_keypoints(in_path, out_path, args.n_features)
        print(f"{in_path.name}: {n_kp} keypoints -> {out_path}")


if __name__ == "__main__":
    main()
