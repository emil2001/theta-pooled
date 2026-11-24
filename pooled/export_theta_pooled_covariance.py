#!/usr/bin/env python3
"""Export a pooled covariance from theta RLE chains for an independent pilot.

The output is a small, name-checked text format intended for an external theta
MCMCStrategy.  Covariance estimation uses raw MCMC multiplicities from
``weight`` and supports an exact additional burn-in in raw proposal steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--burn-in-steps", type=int, default=0)
    return parser.parse_args()


def read_vector(tree: ROOT.TTree, expression: str) -> np.ndarray:
    count = int(tree.Draw(expression, "", "goff"))
    if count != tree.GetEntries():
        raise RuntimeError(
            f"{tree.GetName()}:{expression} read {count}/{tree.GetEntries()}"
        )
    return np.asarray(tree.GetV1())[:count].astype(np.float64, copy=True)


def retained_weights(
    weights: np.ndarray, burn_in_steps: int
) -> tuple[int, np.ndarray]:
    if np.any(weights <= 0) or np.any(weights != np.floor(weights)):
        raise RuntimeError("Theta RLE weights must be positive integers")
    integer_weights = weights.astype(np.int64)
    total = int(integer_weights.sum())
    if burn_in_steps >= total:
        raise ValueError(f"Burn-in {burn_in_steps} is not shorter than {total}")
    if burn_in_steps <= 0:
        return 0, integer_weights
    cumulative = np.cumsum(integer_weights)
    first = int(np.searchsorted(cumulative, burn_in_steps, side="right"))
    kept = integer_weights[first:].copy()
    consumed_before = int(cumulative[first - 1]) if first else 0
    consumed_in_first = burn_in_steps - consumed_before
    if consumed_in_first:
        kept[0] -= consumed_in_first
    return first, kept


def tree_names(root_file: ROOT.TFile) -> list[str]:
    return sorted(
        {
            key.GetName()
            for key in root_file.GetListOfKeys()
            if key.GetName().startswith("chain_")
        },
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def parameter_names(tree: ROOT.TTree) -> list[str]:
    return [
        branch.GetName()
        for branch in tree.GetListOfBranches()
        if branch.GetName() not in {"weight", "nll"}
    ]


def read_parameter_matrix(
    tree: ROOT.TTree, names: list[str], first: int
) -> np.ndarray:
    entries = int(tree.GetEntries())
    tree.SetEstimate(entries + 1)
    result = np.empty((entries - first, len(names)), dtype=np.float64)
    for column_start in range(0, len(names), 4):
        group = names[column_start : column_start + 4]
        count = int(tree.Draw(":".join(group), "", "goff"))
        if count != entries:
            raise RuntimeError(
                f"{tree.GetName()}:{group} read {count}/{entries}"
            )
        buffers = (tree.GetV1(), tree.GetV2(), tree.GetV3(), tree.GetV4())
        for offset, _ in enumerate(group):
            result[:, column_start + offset] = np.asarray(buffers[offset])[
                first:count
            ]
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{tree.GetName()} contains non-finite parameters")
    return result


def combine_moments(
    count: int,
    mean: np.ndarray | None,
    m2: np.ndarray | None,
    new_count: int,
    new_mean: np.ndarray,
    new_m2: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    if mean is None or m2 is None or count == 0:
        return new_count, new_mean, new_m2
    combined = count + new_count
    delta = new_mean - mean
    updated_mean = mean + delta * (new_count / combined)
    updated_m2 = (
        m2
        + new_m2
        + np.outer(delta, delta) * (count * new_count / combined)
    )
    return combined, updated_mean, updated_m2


def write_covariance(
    output: Path,
    names: list[str],
    mean: np.ndarray,
    covariance: np.ndarray,
    metadata: dict[str, object],
) -> None:
    lines = [
        f"theta_pooled_covariance_v1 {len(names)}",
        "parameters " + " ".join(names),
        "mean " + " ".join(f"{value:.17g}" for value in mean),
        "covariance",
    ]
    lines.extend(
        " ".join(f"{value:.17g}" for value in row) for row in covariance
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    if args.burn_in_steps < 0:
        raise ValueError("--burn-in-steps must be non-negative")
    names: list[str] | None = None
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    sources: list[dict[str, object]] = []

    for path in args.inputs:
        root_file = ROOT.TFile.Open(str(path))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Cannot open {path}")
        try:
            for tree_name in tree_names(root_file):
                tree = root_file.Get(tree_name)
                current_names = parameter_names(tree)
                if names is None:
                    names = current_names
                elif current_names != names:
                    raise RuntimeError(
                        f"Parameter order differs in {path}:{tree_name}"
                    )
                tree.SetEstimate(tree.GetEntries() + 1)
                raw_weights = read_vector(tree, "weight")
                first, weights = retained_weights(
                    raw_weights, args.burn_in_steps
                )
                values = read_parameter_matrix(tree, names, first)
                chain_count = int(weights.sum())
                chain_mean = np.average(values, axis=0, weights=weights)
                values -= chain_mean
                values *= np.sqrt(weights)[:, None]
                chain_m2 = values.T @ values
                count, mean, m2 = combine_moments(
                    count,
                    mean,
                    m2,
                    chain_count,
                    chain_mean,
                    chain_m2,
                )
                sources.append(
                    {
                        "file": str(path),
                        "tree": tree_name,
                        "raw_steps": int(raw_weights.sum()),
                        "retained_steps": chain_count,
                        "saved_states": int(tree.GetEntries()),
                    }
                )
        finally:
            root_file.Close()

    if names is None or mean is None or m2 is None or count == 0:
        raise RuntimeError("No theta chains found")
    covariance = m2 / count
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    metadata = {
        "format": "theta_pooled_covariance_v1",
        "inputs": [str(path) for path in args.inputs],
        "additional_burn_in_steps_per_chain": args.burn_in_steps,
        "sources": sources,
        "parameters": names,
        "retained_steps": count,
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(
            eigenvalues.max() / max(eigenvalues.min(), np.finfo(float).tiny)
        ),
    }
    write_covariance(args.output, names, mean, covariance, metadata)
    print(args.output)


if __name__ == "__main__":
    main()
