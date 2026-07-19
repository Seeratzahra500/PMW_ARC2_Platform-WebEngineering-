#!/usr/bin/env python3
"""
Rohtas Fort — Footage to 3D Reconstruction Pipeline
=====================================================
Goal: Convert raw video/photo footage of Rohtas Fort into a dense 3D point
cloud, exported as a .ply file, using COLMAP (Structure-from-Motion + 
Multi-View Stereo).

Pipeline stages:
    1. Frame extraction   (video -> image sequence, via ffmpeg)
    2. Feature extraction (SIFT keypoints per image)
    3. Feature matching   (exhaustive or sequential matcher)
    4. Sparse reconstruction (incremental SfM -> camera poses + sparse 3D points)
    5. Dense reconstruction  (MVS: depth maps -> dense point cloud)
    6. Export .ply           (final textured/colored point cloud)

Usage:
    python3 reconstruct.py --input /path/to/video.mp4
    python3 reconstruct.py --input /path/to/images_folder/
    python3 reconstruct.py --input /path/to/video.mp4 --fps 2 --quality fast
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
FRAMES_DIR = PROJECT_ROOT / "frames"
IMAGES_DIR = PROJECT_ROOT / "images"
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = WORKSPACE_DIR / "database.db"
SPARSE_DIR = WORKSPACE_DIR / "sparse"
DENSE_DIR = WORKSPACE_DIR / "dense"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def log(stage, msg):
    print(f"\n{'='*70}\n[{stage}] {msg}\n{'='*70}")


def run(cmd, stage):
    """Run a shell command, streaming output, and raise on failure."""
    log(stage, "Running: " + " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"[{stage}] failed with exit code {result.returncode}")
    return result


# ---------------------------------------------------------------------------
# Stage 1: Frame extraction
# ---------------------------------------------------------------------------
def extract_frames(input_path: Path, fps: float):
    """If input is a video, sample frames at `fps`. If it's a folder of
    images, just copy/symlink them into IMAGES_DIR."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        log("FRAME EXTRACTION", f"Input is a folder — collecting existing images from {input_path}")
        count = 0
        for f in sorted(input_path.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                shutil.copy(f, IMAGES_DIR / f.name)
                count += 1
        if count == 0:
            raise RuntimeError(f"No images found in {input_path}")
        log("FRAME EXTRACTION", f"Copied {count} images.")
        return count

    # It's a video file
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    log("FRAME EXTRACTION", f"Sampling frames from video at {fps} fps")
    out_pattern = str(FRAMES_DIR / "frame_%05d.jpg")
    run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"fps={fps},scale='min(1600,iw)':-2",
        "-qscale:v", "2",
        out_pattern,
    ], "FRAME EXTRACTION")

    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames — check input video.")
    for f in frames:
        shutil.copy(f, IMAGES_DIR / f.name)
    log("FRAME EXTRACTION", f"Extracted {len(frames)} frames -> {IMAGES_DIR}")
    return len(frames)


# ---------------------------------------------------------------------------
# Stage 2 + 3: Feature extraction & matching (COLMAP)
# ---------------------------------------------------------------------------
def feature_extraction_and_matching(matcher: str):
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    run([
        "colmap", "feature_extractor",
        "--database_path", str(DB_PATH),
        "--image_path", str(IMAGES_DIR),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "0",
    ], "FEATURE EXTRACTION")

    matcher_cmd = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    run([
        "colmap", matcher_cmd,
        "--database_path", str(DB_PATH),
        "--SiftMatching.use_gpu", "0",
    ], "FEATURE MATCHING")


# ---------------------------------------------------------------------------
# Stage 4: Sparse reconstruction (incremental SfM)
# ---------------------------------------------------------------------------
def sparse_reconstruction():
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    run([
        "colmap", "mapper",
        "--database_path", str(DB_PATH),
        "--image_path", str(IMAGES_DIR),
        "--output_path", str(SPARSE_DIR),
    ], "SPARSE RECONSTRUCTION (SfM)")

    models = sorted(SPARSE_DIR.iterdir())
    if not models:
        raise RuntimeError(
            "SfM produced no reconstruction. This usually means too few "
            "overlapping/matchable images — try more frames, better lighting, "
            "or more overlap between shots."
        )
    return models[0]  # model "0" is COLMAP's best/first reconstruction


# ---------------------------------------------------------------------------
# Stage 5: Dense reconstruction (MVS)
# ---------------------------------------------------------------------------
def dense_reconstruction(sparse_model_path: Path, quality: str):
    DENSE_DIR.mkdir(parents=True, exist_ok=True)

    run([
        "colmap", "image_undistorter",
        "--image_path", str(IMAGES_DIR),
        "--input_path", str(sparse_model_path),
        "--output_path", str(DENSE_DIR),
        "--output_type", "COLMAP",
    ], "IMAGE UNDISTORTION")

    window_radius = "3" if quality == "fast" else "5"
    run([
        "colmap", "patch_match_stereo",
        "--workspace_path", str(DENSE_DIR),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.window_radius", window_radius,
    ], "PATCH MATCH STEREO (depth maps)")


# ---------------------------------------------------------------------------
# Stage 6: Export .ply
# ---------------------------------------------------------------------------
def export_ply():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_ply = OUTPUT_DIR / "rohtas_fort_dense.ply"
    run([
        "colmap", "stereo_fusion",
        "--workspace_path", str(DENSE_DIR),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(out_ply),
    ], "STEREO FUSION -> .ply EXPORT")

    if not out_ply.exists():
        raise RuntimeError("stereo_fusion did not produce a .ply file.")
    size_mb = out_ply.stat().st_size / (1024 * 1024)
    log("EXPORT COMPLETE", f"{out_ply} ({size_mb:.1f} MB)")
    return out_ply


# ---------------------------------------------------------------------------
# Also export the sparse point cloud as .ply (fast fallback / preview)
# ---------------------------------------------------------------------------
def export_sparse_ply(sparse_model_path: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_ply = OUTPUT_DIR / "rohtas_fort_sparse.ply"
    run([
        "colmap", "model_converter",
        "--input_path", str(sparse_model_path),
        "--output_path", str(out_ply),
        "--output_type", "PLY",
    ], "SPARSE -> .ply EXPORT")
    return out_ply


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rohtas Fort footage -> 3D .ply reconstruction")
    parser.add_argument("--input", required=True, help="Path to video file OR folder of images")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate if input is video")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential",
                         help="sequential = fast, good for video walk-throughs; exhaustive = for unordered photo sets")
    parser.add_argument("--quality", choices=["fast", "high"], default="fast",
                         help="fast = quicker CPU-only demo; high = slower, denser result")
    parser.add_argument("--skip-dense", action="store_true",
                         help="Stop after sparse reconstruction (much faster, good for a quick demo run)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    t0 = time.time()

    n_images = extract_frames(input_path, args.fps)
    feature_extraction_and_matching(args.matcher)
    sparse_model = sparse_reconstruction()
    sparse_ply = export_sparse_ply(sparse_model)

    print(f"\nSparse point cloud ready: {sparse_ply}")

    if args.skip_dense:
        log("DONE", f"Skipped dense stage. Total time: {time.time()-t0:.1f}s")
        return

    dense_reconstruction(sparse_model, args.quality)
    dense_ply = export_ply()

    log("PIPELINE COMPLETE", (
        f"Input images: {n_images}\n"
        f"Sparse cloud: {sparse_ply}\n"
        f"Dense cloud:  {dense_ply}\n"
        f"Total time:   {time.time()-t0:.1f}s"
    ))


if __name__ == "__main__":
    main()
