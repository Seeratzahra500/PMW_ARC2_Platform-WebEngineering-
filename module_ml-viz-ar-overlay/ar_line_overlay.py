#!/usr/bin/env python3
"""
AR Line Overlay — Architectural Edge Highlighting
====================================================
Goal: Detect and overlay structural lines (walls, arches, gate outlines) onto
a photo — an OpenCV stand-in for the kind of line-based AR overlay used in
heritage-documentation AR apps (highlighting structure edges live over camera
feed). Here it runs on a static photo; the same detect -> draw step is what
would run per-frame in an AR pass.

Pipeline: grayscale -> bilateral filter (denoise, keep edges) -> Canny edges
-> probabilistic Hough transform -> draw lines over the original photo.

Usage:
    python3 ar_line_overlay.py --input photo.jpg --output out.png
    python3 ar_line_overlay.py --input folder_of_photos/ --output-dir viz/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def overlay_lines(img_path: Path, out_path: Path,
                   canny_low: int = 50, canny_high: int = 150,
                   hough_thresh: int = 60, min_line_len: int = 40, max_line_gap: int = 10):
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=60, sigmaSpace=60)
    edges = cv2.Canny(denoised, canny_low, canny_high)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_thresh, minLineLength=min_line_len, maxLineGap=max_line_gap,
    )

    overlay = img.copy()
    n_lines = 0
    if lines is not None:
        n_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (60, 200, 255), 2, cv2.LINE_AA)  # terracotta highlight

    # Blend overlay with original for a semi-transparent AR feel
    vis = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    label = f"{n_lines} structural lines detected"
    cv2.rectangle(vis, (0, 0), (min(vis.shape[1], 480), 40), (30, 24, 20), -1)
    cv2.putText(vis, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (231, 220, 200), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return n_lines


def main():
    parser = argparse.ArgumentParser(description="AR-style structural line overlay")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--output", help="Output image path (single-file mode)")
    parser.add_argument("--output-dir", help="Output folder (batch mode, for a folder input)")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-thresh", type=int, default=60)
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}

    if in_path.is_dir():
        out_dir = Path(args.output_dir or "viz_lines")
        images = sorted([p for p in in_path.iterdir() if p.suffix.lower() in exts])
        if not images:
            print(f"No images found in {in_path}")
            sys.exit(1)
        for img_path in images:
            out_path = out_dir / f"{img_path.stem}_lines.png"
            n = overlay_lines(img_path, out_path, args.canny_low, args.canny_high, args.hough_thresh)
            print(f"{img_path.name}: {n} lines -> {out_path}")
    else:
        out_path = Path(args.output or f"{in_path.stem}_lines.png")
        n = overlay_lines(in_path, out_path, args.canny_low, args.canny_high, args.hough_thresh)
        print(f"{in_path.name}: {n} lines -> {out_path}")


if __name__ == "__main__":
    main()
