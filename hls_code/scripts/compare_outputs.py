#!/usr/bin/env python3

# HOW TO USE
#python3 compare_outputs.py --file conv1d_out.txt --mode numeric --threshold 0.001

import argparse
from pathlib import Path
from typing import List, Tuple, Sequence
import math


def read_lines(path: Path) -> List[str]:
    return [line.rstrip("\n\r") for line in path.read_text().splitlines()]


def is_float_line(line: str) -> bool:
    try:
        float(line.strip())
        return True
    except Exception:
        return False


def compare_numeric_lines(lines1: Sequence[str], lines2: Sequence[str], abs_tol: float, rel_tol: float, max_diffs: int = 20) -> Tuple[bool, dict]:
    min_len = min(len(lines1), len(lines2))
    diffs = []
    sum_sq = 0.0
    count = 0
    for i in range(min_len):
        l1 = lines1[i].strip()
        l2 = lines2[i].strip()
        if l1 == "" and l2 == "":
            continue
        try:
            v1 = float(l1)
            v2 = float(l2)
        except ValueError:
            if l1 != l2:
                diffs.append((i + 1, l1, l2, None, None))
            continue
        diff = v1 - v2
        abs_diff = abs(diff)
        rel_diff = abs_diff / (abs(v2) + 1e-12)
        sum_sq += diff * diff
        count += 1
        if abs_diff > abs_tol and rel_diff > rel_tol:
            if len(diffs) < max_diffs:
                diffs.append((i + 1, v1, v2, abs_diff, rel_diff))
    result = {
        "line_count1": len(lines1),
        "line_count2": len(lines2),
        "numeric_count": count,
        "rms_error": math.sqrt(sum_sq / count) if count > 0 else 0.0,
        "max_abs_diff": max((abs(float(lines1[i].strip()) - float(lines2[i].strip())) for i in range(min_len) if lines1[i].strip() != "" and lines2[i].strip() != ""), default=0.0),
        "diffs": diffs,
    }
    is_same = len(diffs) == 0 and len(lines1) == len(lines2)
    return is_same, result


def compare_files(path1: Path, path2: Path, mode: str, abs_tol: float, rel_tol: float, verbose: bool = False) -> Tuple[str, dict]:
    data1 = path1.read_bytes()
    data2 = path2.read_bytes()
    if data1 == data2:
        return "exact", {"message": "Files are exactly identical (byte-for-byte)."}

    text1 = read_lines(path1)
    text2 = read_lines(path2)

    if mode == "exact":
        return "different", {"message": "Different binary content."}

    if mode == "text":
        if text1 == text2:
            return "same", {"message": "Same line-by-line text."}
        # find first few line differences
        diffs = []
        max_lines = min(len(text1), len(text2))
        for i in range(max_lines):
            if text1[i] != text2[i] and len(diffs) < 20:
                diffs.append((i + 1, text1[i], text2[i]))
        return "different", {
            "message": "Text differs line-by-line.",
            "line_count1": len(text1),
            "line_count2": len(text2),
            "diffs": diffs,
        }

    # numeric mode fallback
    # Handle the common case where one file is a single line of space-separated values
    # (e.g., a flattened vector) and the other is one value per line.
    if len(text1) == 1 and " " in text1[0] and len(text2) > 1:
        text1 = text1[0].strip().split()
    if len(text2) == 1 and " " in text2[0] and len(text1) > 1:
        text2 = text2[0].strip().split()

    all_float = all(is_float_line(x) for x in text1) and all(is_float_line(x) for x in text2)
    if not all_float:
        # report text difference if non-numeric
        if text1 == text2:
            return "same", {"message": "Same line-by-line text."}
        diffs = []
        max_lines = min(len(text1), len(text2))
        for i in range(max_lines):
            if text1[i] != text2[i] and len(diffs) < 20:
                diffs.append((i + 1, text1[i], text2[i]))
        return "different", {
            "message": "Not numeric: text differs.",
            "line_count1": len(text1),
            "line_count2": len(text2),
            "diffs": diffs,
        }

    is_same, metrics = compare_numeric_lines(text1, text2, abs_tol, rel_tol)
    if is_same:
        return "same", {
            "message": f"Numeric compare passed (abs_tol={abs_tol}, rel_tol={rel_tol}).",
            **metrics,
        }

    msg = f"Numeric compare failed: {len(metrics['diffs'])} values exceed tolerances (abs_tol={abs_tol}, rel_tol={rel_tol})."
    if verbose:
        msg += " See diffs in metrics."
    metrics["message"] = msg
    return "different", metrics


def compare_single_file(dir1: Path, dir2: Path, filename: str, mode: str, abs_tol: float, rel_tol: float, verbose: bool = False):
    if not dir1.exists() or not dir1.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir1}")
    if not dir2.exists() or not dir2.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir2}")

    f1 = dir1 / filename
    f2 = dir2 / filename
    if not f1.exists() or not f1.is_file():
        raise FileNotFoundError(f"File not found in {dir1}: {filename}")
    if not f2.exists() or not f2.is_file():
        raise FileNotFoundError(f"File not found in {dir2}: {filename}")

    status, info = compare_files(f1, f2, mode, abs_tol, rel_tol, verbose=verbose)
    if status in ("exact", "same"):
        print(f"[SAME] {filename}: {info.get('message')}")
    else:
        print(f"[DIFF] {filename}: {info.get('message')}")
        if "diffs" in info and info["diffs"]:
            print("All mismatched lines (first 500 shown):")
            for d in info["diffs"][:500]:
                if len(d) >= 5:
                    print(f"  line {d[0]}: {d[1]} vs {d[2]} abs_diff={d[3]} rel_diff={d[4]}")
                else:
                    print(f"  line {d[0]}: {d[1]} vs {d[2]}")
    print(f"\nSummary for file {filename}: status={status}")
    if "line_count1" in info and "line_count2" in info:
        print(f"  lines: {info['line_count1']} vs {info['line_count2']}")
    if "numeric_count" in info:
        print(f"  numeric values compared: {info['numeric_count']}")
    if "rms_error" in info:
        print(f"  RMS error: {info['rms_error']:.6g}")
    if "max_abs_diff" in info:
        print(f"  max abs diff: {info['max_abs_diff']:.6g}")


def main():
    parser = argparse.ArgumentParser(description="Compare one file name across two directories.")
    parser.add_argument("dir1", nargs="?", default="matlab_outputs", help="First directory")
    parser.add_argument("dir2", nargs="?", default="vitis_outputs", help="Second directory")
    parser.add_argument("--file", default="conv1d_out.txt", help="File name to compare")
    parser.add_argument("--mode", choices=["numeric", "text", "exact"], default="numeric", help="Comparison mode")
    parser.add_argument("--abs_tol", type=float, default=1e-6, help="Absolute tolerance for numeric compare")
    parser.add_argument("--rel_tol", type=float, default=1e-6, help="Relative tolerance for numeric compare")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Threshold for overall numeric tolerance (applies to both abs and rel unless explicit values set)")
    parser.add_argument("--verbose", action="store_true", help="Show more detail for same files")
    args = parser.parse_args()

    abs_tol = args.abs_tol
    rel_tol = args.rel_tol
    if args.threshold is not None and args.threshold > 0:
        abs_tol = max(abs_tol, args.threshold)
        rel_tol = max(rel_tol, args.threshold)

    compare_single_file(Path(args.dir1), Path(args.dir2), args.file, args.mode, abs_tol=abs_tol, rel_tol=rel_tol, verbose=args.verbose)


if __name__ == "__main__":
    main()
