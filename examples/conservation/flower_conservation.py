#!/usr/bin/env python3
"""FlowER conservation metric from a prediction file.

FlowER computes conservation *during inference* (see eval_multiGPU.py): every
test reaction emits a leading 5-slot tally on its result line.

Line format:  [A, B, C, D, E]|<seq_idx>|[('SMILES...', count, conserved), ...]

The 5 slots count, over the N samples drawn for that reaction:
  A - correct   : valid SMILES, matches target, electrons conserved
  B - conserved : valid SMILES, wrong product, electrons conserved
  C - non_cons  : valid SMILES, electrons NOT conserved
  D - no_smi_cons   : no valid SMILES, electrons conserved
  E - no_smi_noncons: no valid SMILES, electrons NOT conserved

Conservation metric = (A + B) / N, averaged over all lines: the fraction of
samples that yield a valid molecule AND conserve electrons. Because FlowER
conserves heavy atoms, protons and electrons by construction, this single value
stands in for all of them (contrast g2s_conservation.py, which must recompute
four separate metrics from raw SMILES).

Usage:
    python flower_conservation.py <prediction.txt>
"""

import argparse
import ast
import sys
from pathlib import Path


def parse_array(line):
    """Return the leading [A, B, C, D, E] list of ints, or None."""
    line = line.strip()
    if not line:
        return None
    # The array is everything up to and including the first ']'.
    end = line.find("]")
    if not line.startswith("[") or end == -1:
        return None
    return ast.literal_eval(line[: end + 1])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", type=Path, help="FlowER prediction .txt file")
    args = ap.parse_args()

    if not args.path.is_file():
        sys.exit(f"File not found: {args.path}")

    # STEP 1 - parse each line's [A, B, C, D, E] array.
    arrays = []
    with open(args.path) as fh:
        for line in fh:
            arr = parse_array(line)
            if arr is not None:
                arrays.append(arr)

    if not arrays:
        sys.exit("No parsable lines found.")

    # STEP 2 - per line, (A + B) / N ; STEP 3 - average over lines.
    vals = [(arr[0] + arr[1]) / sum(arr) for arr in arrays if sum(arr) > 0]
    n = len(vals)
    conservation = sum(vals) / n

    # Category totals across the whole file (for transparency).
    labels = ["correct", "conserved", "non_cons", "no_smi_cons", "no_smi_noncons"]
    totals = [sum(arr[i] for arr in arrays) for i in range(5)]

    print(f"lines (reactions) parsed = {n}")
    print(f"samples (sum of arrays)  = {sum(totals)}")
    print("category totals          = "
          + ", ".join(f"{name}={tot}" for name, tot in zip(labels, totals)))
    print(f"conservation (A+B)/N     = {conservation:.6f} = {conservation * 100:.2f}%")


if __name__ == "__main__":
    main()
