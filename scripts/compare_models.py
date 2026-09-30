"""
compare_models.py
=================
Generates a side-by-side comparison table of the DeepLabV3+ and ResNet34-FCN
models for the glacial lake segmentation project.

The script reads:
  1. Model checkpoints — for parameter counts and training history.
  2. Evaluation JSON files produced by ``scripts/evaluate.py`` — for validation
     metrics (precision, recall, F1/Dice, IoU, accuracy).

It produces:
  - A human-readable Markdown table (comparison_table.md)
  - A plain-text table (comparison_table.txt)
  - A machine-readable JSON summary (comparison_summary.json)

All outputs are written to --output-dir (default: results/comparison/).

Usage:
    python scripts/compare_models.py

    # Custom paths
    python scripts/compare_models.py \\
        --dlv3-checkpoint  checkpoints/best_amp_model.pth \\
        --dlv3-eval        results/evaluation/eval_metrics.json \\
        --fcn-checkpoint   resnet34_fcn/checkpoints/best_model.pth \\
        --fcn-eval         resnet34_fcn/results/evaluation/eval_metrics.json \\
        --output-dir       results/comparison

Limitations:
  - Metrics in this table are validation metrics, NOT test-set metrics.
  - The DeepLabV3+ model was trained on Google Colab with CUDA AMP.
    The ResNet34-FCN baseline was trained locally on Apple Silicon MPS
    without AMP. Training-time durations are therefore not directly comparable.
  - Both models were evaluated on the same 2,367-sample validation set
    using identical evaluation code (scripts/evaluate.py).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.resnet34_fcn import ResNet34FCN
from src.utils import ensure_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a comparison table for DeepLabV3+ vs ResNet34-FCN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dlv3-checkpoint",
        type=str,
        default="checkpoints/best_amp_model.pth",
        help="Path to the best DeepLabV3+ checkpoint",
    )
    parser.add_argument(
        "--dlv3-eval",
        type=str,
        default="results/evaluation/eval_metrics.json",
        help="Path to eval_metrics.json produced for DeepLabV3+",
    )
    parser.add_argument(
        "--fcn-checkpoint",
        type=str,
        default="resnet34_fcn/checkpoints/best_model.pth",
        help="Path to the best ResNet34-FCN checkpoint",
    )
    parser.add_argument(
        "--fcn-eval",
        type=str,
        default="resnet34_fcn/results/evaluation/eval_metrics.json",
        help="Path to eval_metrics.json produced for ResNet34-FCN",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory to write comparison table and JSON summary",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Optional[Dict]:
    """Load a JSON file; return None and warn if missing."""
    p = Path(path)
    if not p.is_file():
        print(f"  [MISSING] {p} — skipping (run evaluate.py first)")
        return None
    with open(p) as f:
        return json.load(f)


def _load_checkpoint_meta(chk_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a checkpoint without loading model weights into memory.

    Returns a dict with keys:
      epoch, best_f1, model_name, total_train_seconds, history
    """
    p = Path(chk_path)
    if not p.is_file():
        print(f"  [MISSING] {p}")
        return {}

    print(f"  Reading checkpoint: {p}")
    chk = torch.load(p, map_location="cpu", weights_only=False)

    history = chk.get("history", [])

    # Total training time: sum epoch_time fields when available
    total_train_seconds = None
    if history and isinstance(history[0], dict) and "epoch_time" in history[0]:
        total_train_seconds = sum(r.get("epoch_time", 0.0) for r in history)

    # Best F1: prefer the stored value; fall back to max over history
    best_f1 = chk.get("best_f1", chk.get("best_metric_val"))
    if best_f1 is None and history:
        f1_vals = [r.get("val_f1", 0.0) for r in history if "val_f1" in r]
        best_f1 = max(f1_vals) if f1_vals else None

    # Final val loss
    final_val_loss = None
    if history:
        final_record = history[-1] if isinstance(history[-1], dict) else {}
        final_val_loss = final_record.get("val_loss")

    return {
        "epoch": chk.get("epoch", "?"),
        "best_f1": best_f1,
        "model_name": chk.get("model_name", "unknown"),
        "total_train_seconds": total_train_seconds,
        "final_val_loss": final_val_loss,
        "history_length": len(history),
    }


def _param_count(model_class, **kwargs) -> int:
    """Instantiate model with pretrained=False and count all parameters."""
    m = model_class(pretrained=False, **kwargs)
    return sum(p.numel() for p in m.parameters())


def _fmt_seconds(secs: Optional[float]) -> str:
    """Format seconds as 'Xh Ym Zs' or '—' if unknown."""
    if secs is None:
        return "—"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.2f}%" if v is not None else "—"


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def build_comparison(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    ensure_dirs(str(out_dir))

    print("\n" + "=" * 64)
    print("  Model Comparison: DeepLabV3+ vs ResNet34-FCN")
    print("=" * 64)

    # ----------------------------------------------------------------
    # 1. Parameter counts (instantiate without pretrained weights)
    # ----------------------------------------------------------------
    print("\n[1/4] Computing parameter counts ...")
    dlv3_params = _param_count(DeepLabV3Plus, num_classes=1)
    fcn_params  = _param_count(ResNet34FCN, num_classes=1)

    dlv3_enc = sum(
        p.numel() for p in DeepLabV3Plus(pretrained=False).encoder_parameters()
    )
    fcn_enc = sum(
        p.numel() for p in ResNet34FCN(pretrained=False).encoder_parameters()
    )

    # ----------------------------------------------------------------
    # 2. Checkpoint metadata (training history, timing)
    # ----------------------------------------------------------------
    print("\n[2/4] Loading checkpoint metadata ...")
    dlv3_meta = _load_checkpoint_meta(args.dlv3_checkpoint)
    fcn_meta  = _load_checkpoint_meta(args.fcn_checkpoint)

    # ----------------------------------------------------------------
    # 3. Evaluation metrics
    # ----------------------------------------------------------------
    print("\n[3/4] Loading evaluation results ...")
    dlv3_eval = _load_json(args.dlv3_eval)
    fcn_eval  = _load_json(args.fcn_eval)

    def _m(eval_json: Optional[Dict], key: str) -> Optional[float]:
        if eval_json is None:
            return None
        return eval_json.get("metrics", {}).get(key)

    def _cfg(eval_json: Optional[Dict], key: str) -> Any:
        if eval_json is None:
            return None
        return eval_json.get("evaluation_config", {}).get(key)

    # ----------------------------------------------------------------
    # 4. Assemble comparison data
    # ----------------------------------------------------------------
    print("\n[4/4] Building comparison table ...")

    rows = [
        # (Row label, DeepLabV3+ value string, ResNet34-FCN value string, note)
        ("Architecture",
         "DeepLabV3+ (ResNet-50 + ASPP)",
         "ResNet34-FCN (ResNet-34 + FCN decoder)",
         ""),
        ("Encoder backbone",
         "ResNet-50",
         "ResNet-34",
         "Both ImageNet-pretrained"),
        ("Decoder",
         "ASPP + low-level feature fusion",
         "5-stage progressive transposed-conv",
         ""),
        ("Total parameters",
         f"{dlv3_params:,}",
         f"{fcn_params:,}",
         "ResNet34-FCN is ~40% smaller"),
        ("Encoder parameters",
         f"{dlv3_enc:,}",
         f"{fcn_enc:,}",
         ""),
        ("Decoder parameters",
         f"{dlv3_params - dlv3_enc:,}",
         f"{fcn_params - fcn_enc:,}",
         ""),
        ("Training device",
         "Google Colab GPU (CUDA + AMP)",
         "Apple Silicon MPS (no AMP)",
         "Not directly comparable"),
        ("Training epochs",
         f"{dlv3_meta.get('epoch', '?')}",
         f"{fcn_meta.get('epoch', '?')}",
         ""),
        ("Total training time",
         _fmt_seconds(dlv3_meta.get("total_train_seconds")),
         _fmt_seconds(fcn_meta.get("total_train_seconds")),
         "Wall-clock time on respective hardware"),
        ("Final val loss",
         _fmt(dlv3_meta.get("final_val_loss"), digits=4),
         _fmt(fcn_meta.get("final_val_loss"), digits=4),
         "BCE+Dice combined loss"),
        ("Val samples evaluated",
         str(_cfg(dlv3_eval, "n_val_samples") or "—"),
         str(_cfg(fcn_eval,  "n_val_samples") or "—"),
         "Same split: data/raw/val"),
        ("Threshold",
         str(_cfg(dlv3_eval, "threshold") or "0.5"),
         str(_cfg(fcn_eval,  "threshold") or "0.5"),
         "Sigmoid output threshold"),
        ("Precision",
         _pct(_m(dlv3_eval, "precision")),
         _pct(_m(fcn_eval,  "precision")),
         "Global pixel-level"),
        ("Recall",
         _pct(_m(dlv3_eval, "recall")),
         _pct(_m(fcn_eval,  "recall")),
         "Global pixel-level"),
        ("F1 / Dice",
         _pct(_m(dlv3_eval, "f1")),
         _pct(_m(fcn_eval,  "f1")),
         "Global pixel-level ← primary metric"),
        ("IoU (Jaccard)",
         _pct(_m(dlv3_eval, "iou")),
         _pct(_m(fcn_eval,  "iou")),
         "Global pixel-level"),
        ("Pixel Accuracy",
         _pct(_m(dlv3_eval, "accuracy")),
         _pct(_m(fcn_eval,  "accuracy")),
         "Global pixel-level"),
    ]

    # ----------------------------------------------------------------
    # 5. Print table to terminal
    # ----------------------------------------------------------------
    col_w = [32, 36, 38]
    sep = "+" + "-" * col_w[0] + "+" + "-" * col_w[1] + "+" + "-" * col_w[2] + "+"

    def _row(a, b, c):
        return f"| {a:<{col_w[0]-2}} | {b:<{col_w[1]-2}} | {c:<{col_w[2]-2}} |"

    print()
    print(sep)
    print(_row("Metric / Property", "DeepLabV3+", "ResNet34-FCN"))
    print(sep)
    for label, dlv3_val, fcn_val, _ in rows:
        print(_row(label, dlv3_val, fcn_val))
    print(sep)

    # Computed delta for key metrics
    print()
    f1_dlv3 = _m(dlv3_eval, "f1")
    f1_fcn  = _m(fcn_eval,  "f1")
    if f1_dlv3 is not None and f1_fcn is not None:
        delta_f1  = (f1_dlv3 - f1_fcn) * 100
        delta_iou = (_m(dlv3_eval, "iou") - _m(fcn_eval, "iou")) * 100
        print(f"  DeepLabV3+ advantage — F1: {delta_f1:+.2f}pp | IoU: {delta_iou:+.2f}pp")
        print(f"  Parameter overhead   : {dlv3_params - fcn_params:+,} params")

    # ----------------------------------------------------------------
    # 6. Write Markdown
    # ----------------------------------------------------------------
    md_path = out_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Model Comparison: DeepLabV3+ vs ResNet34-FCN\n\n")
        f.write(
            "> **Scope**: Validation metrics only. These results are from the\n"
            "> same 2,367-sample validation split used during training.\n"
            "> They are **not** independent test-set results.\n\n"
        )
        f.write("> **Hardware note**: DeepLabV3+ was trained on Google Colab GPU\n")
        f.write("> with CUDA AMP. ResNet34-FCN was trained on Apple Silicon MPS\n")
        f.write("> without AMP. Training-time durations are not directly comparable.\n\n")
        f.write("| Metric / Property | DeepLabV3+ | ResNet34-FCN | Notes |\n")
        f.write("|---|---|---|---|\n")
        for label, dlv3_val, fcn_val, note in rows:
            f.write(f"| {label} | {dlv3_val} | {fcn_val} | {note} |\n")
        f.write("\n")
        if f1_dlv3 is not None and f1_fcn is not None:
            f.write("## Summary\n\n")
            f.write(f"- **F1 difference**: DeepLabV3+ leads by **{delta_f1:+.2f} percentage points**\n")
            f.write(f"- **IoU difference**: DeepLabV3+ leads by **{delta_iou:+.2f} percentage points**\n")
            f.write(f"- **Parameter count**: DeepLabV3+ has **{dlv3_params - fcn_params:,}** more parameters "
                    f"({(dlv3_params / fcn_params - 1) * 100:.1f}% larger)\n\n")
        f.write("## Methodology Notes\n\n")
        f.write("1. Metrics are computed from the **global confusion matrix** accumulated\n")
        f.write("   over all validation pixels — not as an average of per-image scores.\n")
        f.write("2. The comparison uses a sigmoid threshold of **0.5** for both models.\n")
        f.write("3. Both models were trained with the **same loss** (BCE + Dice, 0.5/0.5),\n")
        f.write("   **same optimizer** (AdamW, lr=1e-4), and **same dataset split**.\n")
        f.write("4. The ResNet34-FCN backbone was pretrained on ImageNet\n")
        f.write("   (`ResNet34_Weights.DEFAULT`), not on the GID dataset as described\n")
        f.write("   in Ma et al. (2025), because GID pretrained weights are not publicly available.\n")
        f.write("5. This is not a reproduction of the original paper's experiment.\n")
        f.write("   The comparison is illustrative of the accuracy–complexity trade-off\n")
        f.write("   between the two architectures under our training conditions.\n")
    print(f"\nMarkdown table : {md_path}")

    # ----------------------------------------------------------------
    # 7. Write plain text table
    # ----------------------------------------------------------------
    txt_path = out_dir / "comparison_table.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  Deep Learning Based Glacial Lake Detection and Segmentation\n")
        f.write("  Model Comparison: DeepLabV3+ vs ResNet34-FCN Baseline\n")
        f.write("=" * 80 + "\n\n")
        f.write("IMPORTANT: All metrics are validation metrics (data/raw/val, 2,367 samples).\n")
        f.write("These are NOT independent test-set results.\n\n")
        f.write(f"{'Metric / Property':<34} {'DeepLabV3+':>20}   {'ResNet34-FCN':>20}\n")
        f.write("-" * 80 + "\n")
        for label, dlv3_val, fcn_val, _ in rows:
            f.write(f"  {label:<32} {dlv3_val:>20}   {fcn_val:>20}\n")
        f.write("-" * 80 + "\n")
        if f1_dlv3 is not None and f1_fcn is not None:
            f.write(f"\nDeepLabV3+ F1 advantage : {delta_f1:+.2f} percentage points\n")
            f.write(f"DeepLabV3+ IoU advantage: {delta_iou:+.2f} percentage points\n")
            f.write(f"Extra parameters (DLv3+): {dlv3_params - fcn_params:,}\n")
    print(f"Text table     : {txt_path}")

    # ----------------------------------------------------------------
    # 8. Write JSON summary
    # ----------------------------------------------------------------
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": "validation only — not an independent test set",
        "val_split": "data/raw/val",
        "deeplabv3plus": {
            "checkpoint": args.dlv3_checkpoint,
            "total_parameters": dlv3_params,
            "encoder_parameters": dlv3_enc,
            "decoder_parameters": dlv3_params - dlv3_enc,
            "training": dlv3_meta,
            "validation_metrics": dlv3_eval.get("metrics") if dlv3_eval else None,
            "confusion_matrix": dlv3_eval.get("confusion_matrix") if dlv3_eval else None,
        },
        "resnet34_fcn": {
            "checkpoint": args.fcn_checkpoint,
            "total_parameters": fcn_params,
            "encoder_parameters": fcn_enc,
            "decoder_parameters": fcn_params - fcn_enc,
            "training": fcn_meta,
            "validation_metrics": fcn_eval.get("metrics") if fcn_eval else None,
            "confusion_matrix": fcn_eval.get("confusion_matrix") if fcn_eval else None,
        },
        "delta": {
            "f1_percentage_points": round((f1_dlv3 - f1_fcn) * 100, 4)
            if (f1_dlv3 is not None and f1_fcn is not None) else None,
            "iou_percentage_points": round(
                (_m(dlv3_eval, "iou") - _m(fcn_eval, "iou")) * 100, 4
            ) if (dlv3_eval and fcn_eval) else None,
            "extra_params_deeplabv3plus": dlv3_params - fcn_params,
        },
    }

    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"JSON summary   : {json_path}")

    print(f"\nAll comparison outputs written to: {out_dir.resolve()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    build_comparison(args)
