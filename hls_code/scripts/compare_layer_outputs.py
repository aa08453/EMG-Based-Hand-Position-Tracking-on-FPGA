#!/usr/bin/env python3
"""Compare desired layer outputs against HLS outputs.

This script compares matching text files across two directories and reports where
values differ, by how much, and summary metrics per file.

Usage:
  python3 compare_layer_outputs.py
  python3 compare_layer_outputs.py --ref-dir ../verification_data --hls-dir ../vitis_outputs
  python3 compare_layer_outputs.py --ref-dir ../verification_data --hls-dir ../../updated_hls_code/hls_code/vitis_outputs
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_float_lines(path: Path) -> List[float]:
    text = path.read_text().strip()
    if text == "":
        return []

    lines = text.splitlines()
    values: List[float] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue

        parts = stripped.split()
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                raise ValueError(f"Non-numeric value in {path}: '{part}'")
    return values


def compare_arrays(expected: List[float], actual: List[float]) -> dict:
    n_expected = len(expected)
    n_actual = len(actual)
    n_compare = min(n_expected, n_actual)

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    sum_sq_diff = 0.0
    diffs: List[Tuple[int, float, float, float]] = []

    for idx in range(n_compare):
        exp_val = expected[idx]
        act_val = actual[idx]
        diff = act_val - exp_val
        abs_diff = abs(diff)
        sum_abs_diff += abs_diff
        sum_sq_diff += diff * diff
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
        if abs_diff != 0.0 and len(diffs) < 20:
            diffs.append((idx, exp_val, act_val, abs_diff))

    rms_error = math.sqrt(sum_sq_diff / n_compare) if n_compare else 0.0
    mean_abs_diff = sum_abs_diff / n_compare if n_compare else 0.0

    return {
        "n_expected": n_expected,
        "n_actual": n_actual,
        "n_compare": n_compare,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "rms_error": rms_error,
        "diff_count": len([1 for i in range(n_compare) if abs(actual[i] - expected[i]) > 0.0]),
        "first_diffs": diffs,
    }


def compare_file_pair(ref_path: Path, hls_path: Path, abs_tol: float, rel_tol: float) -> Tuple[str, dict]:
    expected = parse_float_lines(ref_path)
    actual = parse_float_lines(hls_path)
    metrics = compare_arrays(expected, actual)

    tolerant_diffs = []
    n_tolerant = 0
    n_compare = metrics["n_compare"]
    for idx in range(n_compare):
        exp_val = expected[idx]
        act_val = actual[idx]
        abs_diff = abs(act_val - exp_val)
        rel_diff = abs_diff / (abs(exp_val) + 1e-12)
        if abs_diff > abs_tol and rel_diff > rel_tol:
            n_tolerant += 1
            if len(tolerant_diffs) < 20:
                tolerant_diffs.append((idx, exp_val, act_val, abs_diff, rel_diff))

    metrics["tolerant_diff_count"] = n_tolerant
    metrics["tolerant_diffs"] = tolerant_diffs
    exact_lengths = metrics["n_expected"] == metrics["n_actual"]
    pass_numeric = n_tolerant == 0 and exact_lengths

    if pass_numeric:
        status = "PASS"
    else:
        status = "FAIL"

    if metrics["n_expected"] != metrics["n_actual"]:
        status = "FAIL"

    return status, metrics


def list_text_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])


def print_summary(results: List[Tuple[str, Path, Path, str, dict]], extra_ref: List[str], extra_hls: List[str]) -> None:
    print("\n=== Comparison Summary ===")
    passes = sum(1 for _, _, _, status, _ in results if status == "PASS")
    fails = sum(1 for _, _, _, status, _ in results if status != "PASS")
    print(f"Files compared: {len(results)}")
    print(f"  PASS: {passes}")
    print(f"  FAIL: {fails}")
    if extra_ref:
        print(f"Reference-only files ({len(extra_ref)}): {', '.join(sorted(extra_ref))}")
    if extra_hls:
        print(f"HLS-only files ({len(extra_hls)}): {', '.join(sorted(extra_hls))}")

    if fails > 0:
        print("\nFailing files:")
        for filename, _, _, status, metrics in results:
            if status != "PASS":
                print(f"  {filename}: diffs={metrics['tolerant_diff_count']} max_abs_diff={metrics['max_abs_diff']:.6g} n_expected={metrics['n_expected']} n_actual={metrics['n_actual']}")


def run_compare(ref_dir: Path, hls_dir: Path, abs_tol: float, rel_tol: float) -> int:
    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
    if not hls_dir.exists() or not hls_dir.is_dir():
        raise FileNotFoundError(f"HLS output directory not found: {hls_dir}")

    ref_files = {p.name: p for p in list_text_files(ref_dir)}
    hls_files = {p.name: p for p in list_text_files(hls_dir)}
    common = sorted(ref_files.keys() & hls_files.keys())
    extra_ref = sorted(set(ref_files.keys()) - set(hls_files.keys()))
    extra_hls = sorted(set(hls_files.keys()) - set(ref_files.keys()))

    results = []
    for filename in common:
        ref_path = ref_files[filename]
        hls_path = hls_files[filename]
        status, metrics = compare_file_pair(ref_path, hls_path, abs_tol, rel_tol)
        print(f"\n--- {filename} ---")
        print(f"Reference: {ref_path}")
        print(f"HLS output: {hls_path}")
        print(f"Status: {status}")
        print(f"  reference length: {metrics['n_expected']}")
        print(f"  hls length:       {metrics['n_actual']}")
        print(f"  compared values:  {metrics['n_compare']}")
        print(f"  max abs diff:     {metrics['max_abs_diff']:.6g}")
        print(f"  mean abs diff:    {metrics['mean_abs_diff']:.6g}")
        print(f"  RMS error:        {metrics['rms_error']:.6g}")
        print(f"  count > tolerances: {metrics['tolerant_diff_count']}")

        if metrics['tolerant_diff_count'] > 0:
            print("  First differences:")
            for idx, exp_val, act_val, abs_diff, rel_diff in metrics['tolerant_diffs']:
                print(f"    idx={idx} expected={exp_val:.8g} hls={act_val:.8g} abs_diff={abs_diff:.8g} rel_diff={rel_diff:.6g}")

        if metrics['n_expected'] != metrics['n_actual']:
            print("  Warning: file lengths differ and only the overlapping values were compared.")

        results.append((filename, ref_path, hls_path, status, metrics))

    print_summary(results, extra_ref, extra_hls)
    return 0 if all(status == "PASS" for _, _, _, status, _ in results) else 1


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_ref = script_dir.parent / "verification_data"
    default_hls = script_dir.parent / "vitis_outputs"

    parser = argparse.ArgumentParser(description="Compare layer outputs between reference and HLS-produced output files.")
    parser.add_argument("--ref-dir", type=Path, default=default_ref, help="Directory with desired/reference output files")
    parser.add_argument("--hls-dir", type=Path, default=default_hls, help="Directory with HLS-generated output files")
    parser.add_argument("--abs-tol", type=float, default=1e-6, help="Absolute tolerance for numeric comparisons")
    parser.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for numeric comparisons")
    args = parser.parse_args()

    return run_compare(args.ref_dir, args.hls_dir, args.abs_tol, args.rel_tol)


if __name__ == "__main__":
    raise SystemExit(main())
