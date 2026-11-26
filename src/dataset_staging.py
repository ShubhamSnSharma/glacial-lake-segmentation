"""
dataset_staging.py
==================
Canonical dataset discovery, duplicate resolution, and fail-closed staging
for Google Colab and local training environments.

Features:
  1. Identifies canonical sample IDs by recognizing duplicate suffixes like ' (1)', ' (2)'.
  2. Compares SHA-256 byte hashes of duplicate candidates to confirm identical content.
  3. Fails closed with an explicit error if duplicate candidates have conflicting content.
  4. Pairs canonical images and masks strictly by sample ID (not sorted index).
  5. Reports canonical pairs, confirmed duplicates, conflicting duplicates, and unmatched files.
  6. Stages verified canonical pairs into target storage without modifying source directories.
  7. Exhaustively validates image decoding, 512x512 dimensions, and {0, 1, 255} mask values.
"""

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def get_canonical_id(stem: str) -> str:
    """
    Extract the canonical sample ID by stripping duplicate copy suffixes such as ' (1)', ' (2)'.

    Examples:
      '10021 (1)' -> '10021'
      '13607 (2)' -> '13607'
      '10 (1)'    -> '10'
      '10021'     -> '10021'
    """
    return re.sub(r"\s*\(\d+\)$", "", stem.strip())


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of a file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_files_by_canonical_id(directory: Path) -> Dict[str, List[Path]]:
    """Group all PNG files in a directory by their canonical sample ID."""
    groups: Dict[str, List[Path]] = {}
    for p in sorted(directory.glob("*.png")):
        if p.name.startswith("."):
            continue
        cid = get_canonical_id(p.stem)
        groups.setdefault(cid, []).append(p)
    return groups


def resolve_canonical_files(
    directory: Path,
    category_name: str,
) -> Tuple[Dict[str, Path], List[Tuple[str, Path, Path]], List[Tuple[str, List[Path]]]]:
    """
    Resolve files for a single modality (images or labels).

    Returns:
      canonical_map: Dict[canonical_id -> resolved Path]
      confirmed_duplicates: List of (canonical_id, duplicate_path, canonical_path)
      conflicts: List of (canonical_id, [conflicting_paths])
    """
    groups = group_files_by_canonical_id(directory)
    canonical_map: Dict[str, Path] = {}
    confirmed_duplicates: List[Tuple[str, Path, Path]] = []
    conflicts: List[Tuple[str, List[Path]]] = []

    for cid, paths in groups.items():
        if len(paths) == 1:
            canonical_map[cid] = paths[0]
        else:
            # Multiple candidates exist: compute SHA-256 hashes to verify content equivalence
            hashes = [compute_file_sha256(p) for p in paths]
            unique_hashes = set(hashes)

            if len(unique_hashes) > 1:
                # Conflicting content: fail closed
                conflicts.append((cid, paths))
            else:
                # All candidates have identical byte content
                # Prefer unsuffixed name (e.g. '10021.png') if present, else first
                unsuffixed = [p for p in paths if p.stem == cid]
                chosen = unsuffixed[0] if unsuffixed else paths[0]
                canonical_map[cid] = chosen
                for p in paths:
                    if p != chosen:
                        confirmed_duplicates.append((cid, p, chosen))

    return canonical_map, confirmed_duplicates, conflicts


def analyze_split_canonicalization(split_dir: Path, split_name: str) -> Dict[str, Any]:
    """
    Analyze image and label directories in a split, verifying duplicates, pairing by ID,
    and categorizing all files.
    """
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory in {split_name}: '{img_dir}'")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory in {split_name}: '{lbl_dir}'")

    img_canon, img_dups, img_conflicts = resolve_canonical_files(img_dir, f"{split_name}/images")
    lbl_canon, lbl_dups, lbl_conflicts = resolve_canonical_files(lbl_dir, f"{split_name}/labels")

    # Check for content conflicts
    all_conflicts = img_conflicts + lbl_conflicts
    if all_conflicts:
        conflict_msg = "; ".join(
            f"ID '{cid}': {[p.name for p in paths]}" for cid, paths in all_conflicts[:5]
        )
        raise ValueError(
            f"Fail-Closed Check Failed: Conflicting duplicate candidates detected in {split_name}! "
            f"Files with identical canonical ID have differing content: {conflict_msg}"
        )

    # Pair by canonical ID
    img_ids = set(img_canon.keys())
    lbl_ids = set(lbl_canon.keys())
    matched_ids = sorted(img_ids & lbl_ids)

    unmatched_images = sorted(img_ids - lbl_ids)
    unmatched_labels = sorted(lbl_ids - img_ids)

    canonical_pairs = [(img_canon[cid], lbl_canon[cid], cid) for cid in matched_ids]

    raw_img_count = sum(1 for p in img_dir.glob("*.png") if not p.name.startswith("."))
    raw_lbl_count = sum(1 for p in lbl_dir.glob("*.png") if not p.name.startswith("."))

    return {
        "split_name": split_name,
        "raw_images_count": raw_img_count,
        "raw_labels_count": raw_lbl_count,
        "canonical_images_count": len(img_canon),
        "canonical_labels_count": len(lbl_canon),
        "canonical_pairs": canonical_pairs,
        "confirmed_image_duplicates": img_dups,
        "confirmed_label_duplicates": lbl_dups,
        "conflicting_duplicates": all_conflicts,
        "unmatched_canonical_images": unmatched_images,
        "unmatched_canonical_labels": unmatched_labels,
    }


def find_source_splits(source_root: Path) -> Tuple[Path, Path]:
    """
    Search for valid GLID and val source directories under source_root.
    Checks:
      1. source_root / 'GLID' and source_root / 'val'
      2. source_root / 'data' / 'raw' / 'GLID' and '.../val'
    """
    candidates = [
        (source_root / "GLID", source_root / "val"),
        (source_root / "data" / "raw" / "GLID", source_root / "data" / "raw" / "val"),
    ]
    for train_cand, val_cand in candidates:
        if (train_cand / "images").is_dir() and (val_cand / "images").is_dir():
            return train_cand, val_cand

    raise FileNotFoundError(
        f"Could not locate GLID and val source directories with images/ and labels/ under '{source_root}'. "
        f"Searched: {[str(c[0]) for c in candidates]}"
    )


def stage_split_pairs(
    canonical_pairs: List[Tuple[Path, Path, str]],
    target_split_dir: Path,
    desc: str = "Staging",
    verify_existing_sha256: bool = True,
) -> List[Tuple[Path, Path]]:
    """
    Stage verified canonical pairs to target_split_dir/images and target_split_dir/labels
    using clean canonical filenames ('{cid}.png').

    Safe staging rules:
      1. Destination files are never trusted by size alone. If a destination file exists,
         its SHA-256 is verified against the source file. If hashes differ (or file is corrupted),
         it is safely overwritten.
      2. Stale files from previous staging attempts (e.g. uncanonical suffixed files or orphan files)
         in target_img_dir and target_lbl_dir are safely cleaned up.
      3. Google Drive source files are strictly read-only and never deleted or modified.
    """
    target_img_dir = target_split_dir / "images"
    target_lbl_dir = target_split_dir / "labels"
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    expected_img_names = {f"{cid}.png" for _, _, cid in canonical_pairs}
    expected_lbl_names = {f"{cid}.png" for _, _, cid in canonical_pairs}

    # Clean stale files owned by this staging process in target directories
    for p in target_img_dir.glob("*.png"):
        if not p.name.startswith(".") and p.name not in expected_img_names:
            p.unlink()

    for p in target_lbl_dir.glob("*.png"):
        if not p.name.startswith(".") and p.name not in expected_lbl_names:
            p.unlink()

    staged_pairs: List[Tuple[Path, Path]] = []

    for img_path, lbl_path, cid in tqdm(canonical_pairs, desc=desc, unit="pair"):
        dest_img = target_img_dir / f"{cid}.png"
        dest_lbl = target_lbl_dir / f"{cid}.png"

        # Safe copy or SHA-256 verified reuse for image
        need_img_copy = True
        if dest_img.exists():
            if verify_existing_sha256:
                try:
                    if dest_img.stat().st_size == img_path.stat().st_size:
                        if compute_file_sha256(dest_img) == compute_file_sha256(img_path):
                            need_img_copy = False
                except Exception:
                    need_img_copy = True
            else:
                need_img_copy = True

        if need_img_copy:
            shutil.copy2(img_path, dest_img)

        # Safe copy or SHA-256 verified reuse for label
        need_lbl_copy = True
        if dest_lbl.exists():
            if verify_existing_sha256:
                try:
                    if dest_lbl.stat().st_size == lbl_path.stat().st_size:
                        if compute_file_sha256(dest_lbl) == compute_file_sha256(lbl_path):
                            need_lbl_copy = False
                except Exception:
                    need_lbl_copy = True
            else:
                need_lbl_copy = True

        if need_lbl_copy:
            shutil.copy2(lbl_path, dest_lbl)

        staged_pairs.append((dest_img, dest_lbl))

    return staged_pairs


def verify_staged_split_integrity(
    split_dir: Path,
    expected_pairs: List[Tuple[Path, Path, str]],
    split_name: str,
    expected_count: int,
) -> None:
    """
    Independently enumerate the staged image and label directories.
    Requires:
      1. Exactly expected_count canonical images and expected_count canonical labels.
      2. No extra, missing, or unmatched files.
      3. Verifies that staged content corresponds to the selected canonical source files (SHA-256).
    """
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not (img_dir.is_dir() and lbl_dir.is_dir()):
        raise FileNotFoundError(f"Missing images/ or labels/ in staged split: {split_dir}")

    staged_img_files = {p.stem: p for p in img_dir.glob("*.png") if not p.name.startswith(".")}
    staged_lbl_files = {p.stem: p for p in lbl_dir.glob("*.png") if not p.name.startswith(".")}

    expected_ids = {cid for _, _, cid in expected_pairs}
    src_img_map = {cid: p for p, _, cid in expected_pairs}
    src_lbl_map = {cid: p for _, p, cid in expected_pairs}

    # 1. Independent count verification
    if len(staged_img_files) != expected_count:
        raise ValueError(
            f"Staged Integrity Error [{split_name}]: Found {len(staged_img_files):,} staged images, "
            f"expected exactly {expected_count:,}."
        )
    if len(staged_lbl_files) != expected_count:
        raise ValueError(
            f"Staged Integrity Error [{split_name}]: Found {len(staged_lbl_files):,} staged labels, "
            f"expected exactly {expected_count:,}."
        )

    # 2. Check for unmatched, missing, or extra files
    extra_img = set(staged_img_files.keys()) - expected_ids
    missing_img = expected_ids - set(staged_img_files.keys())
    extra_lbl = set(staged_lbl_files.keys()) - expected_ids
    missing_lbl = expected_ids - set(staged_lbl_files.keys())

    if extra_img or missing_img or extra_lbl or missing_lbl:
        raise ValueError(
            f"Staged Integrity Error [{split_name}]: Staged directory mismatch! "
            f"Extra images: {len(extra_img)}, Missing images: {len(missing_img)}, "
            f"Extra labels: {len(extra_lbl)}, Missing labels: {len(missing_lbl)}."
        )

    unmatched = set(staged_img_files.keys()) ^ set(staged_lbl_files.keys())
    if unmatched:
        raise ValueError(
            f"Staged Integrity Error [{split_name}]: Found {len(unmatched)} unmatched staged pairs!"
        )

    # 3. Verify staged content corresponds to selected canonical source files
    for cid in tqdm(sorted(expected_ids), desc=f"Verifying staged content [{split_name}]", unit="pair"):
        dest_img = staged_img_files[cid]
        src_img = src_img_map[cid]
        dest_lbl = staged_lbl_files[cid]
        src_lbl = src_lbl_map[cid]

        if dest_img.stat().st_size != src_img.stat().st_size:
            raise ValueError(
                f"Staged Content Error [{split_name}]: Staged image '{dest_img.name}' size ({dest_img.stat().st_size}) "
                f"does not match source file size ({src_img.stat().st_size})."
            )
        if dest_lbl.stat().st_size != src_lbl.stat().st_size:
            raise ValueError(
                f"Staged Content Error [{split_name}]: Staged label '{dest_lbl.name}' size ({dest_lbl.stat().st_size}) "
                f"does not match source file size ({src_lbl.stat().st_size})."
            )

        if compute_file_sha256(dest_img) != compute_file_sha256(src_img):
            raise ValueError(
                f"Staged Content Error [{split_name}]: Staged image '{dest_img.name}' SHA-256 mismatch with source!"
            )
        if compute_file_sha256(dest_lbl) != compute_file_sha256(src_lbl):
            raise ValueError(
                f"Staged Content Error [{split_name}]: Staged label '{dest_lbl.name}' SHA-256 mismatch with source!"
            )


def verify_staged_dataset_exhaustively(
    staged_root: Path,
    expected_train: int = 16000,
    expected_val: int = 2367,
) -> Dict[str, Any]:
    """
    Exhaustively verify every image and mask in the staged dataset.
    Checks:
      - Total counts match expected_train and expected_val exactly.
      - 0 orphan files.
      - Image decodes, mask decodes.
      - Exact 512x512 dimensions.
      - Mask values strictly in {0, 1, 255}.
    """
    results: Dict[str, Any] = {}

    for split_name, expected_count in [("GLID", expected_train), ("val", expected_val)]:
        split_dir = staged_root / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not (img_dir.is_dir() and lbl_dir.is_dir()):
            raise FileNotFoundError(f"Missing images/ or labels/ in staged split: {split_dir}")

        img_files = {p.stem: p for p in img_dir.glob("*.png") if not p.name.startswith(".")}
        lbl_files = {p.stem: p for p in lbl_dir.glob("*.png") if not p.name.startswith(".")}

        matched = sorted(img_files.keys() & lbl_files.keys())
        unmatched_img = set(img_files.keys()) - set(lbl_files.keys())
        unmatched_lbl = set(lbl_files.keys()) - set(img_files.keys())

        if unmatched_img or unmatched_lbl:
            raise ValueError(
                f"Fail-Closed Staging Error: Staged split '{split_name}' contains unmatched files! "
                f"Unmatched images: {len(unmatched_img)}, Unmatched labels: {len(unmatched_lbl)}"
            )

        if len(matched) != expected_count:
            raise ValueError(
                f"Fail-Closed Staging Error: Staged split '{split_name}' has {len(matched):,} pairs, "
                f"expected exactly {expected_count:,}."
            )

        has_foreground = 0
        all_background = 0
        observed_encodings: Set[str] = set()

        for cid in tqdm(matched, desc=f"Exhaustive check on staged {split_name}", unit="pair"):
            im_path = img_files[cid]
            lb_path = lbl_files[cid]

            try:
                with Image.open(im_path) as im:
                    im_size = im.size
                    im.load()
            except Exception as e:
                raise ValueError(f"Corrupt staged image '{im_path.name}' in {split_name}: {e}")

            if im_size != (512, 512):
                raise ValueError(f"Invalid staged image size {im_size} in '{im_path.name}' (expected 512x512)")

            try:
                with Image.open(lb_path) as lb:
                    lb_size = lb.size
                    lb_np = np.array(lb)
            except Exception as e:
                raise ValueError(f"Corrupt staged mask '{lb_path.name}' in {split_name}: {e}")

            if lb_size != (512, 512):
                raise ValueError(f"Invalid staged mask size {lb_size} in '{lb_path.name}' (expected 512x512)")

            unique_vals = set(np.unique(lb_np).tolist())
            unexpected = unique_vals - {0, 1, 255}
            if unexpected:
                raise ValueError(
                    f"Unexpected mask pixel values {sorted(unexpected)} in staged mask '{lb_path.name}' (allowed: {{0, 1, 255}})"
                )

            if 1 in unique_vals:
                observed_encodings.add("0/1")
            if 255 in unique_vals:
                observed_encodings.add("0/255")

            if any(v > 0 for v in unique_vals):
                has_foreground += 1
            else:
                all_background += 1

        results[split_name] = {
            "total_pairs": len(matched),
            "foreground_masks": has_foreground,
            "background_masks": all_background,
            "encodings": sorted(observed_encodings),
        }

    return results
