
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Iterator, Sequence

from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from soft_irr.evaluation.distances.base import BipartiteMatchingState

MATRIX_KEY = ("dataset", "negative_mode", "measure", "agreement_level", "seed", "sample_index")


def hungarian_assignment(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]


def greedy_assignment(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    return BipartiteMatchingState._greedy_assignment(cost)


def is_decisive(cost: Sequence[Sequence[float]], tolerance: float = 1e-9) -> bool:
    optimum = hungarian_assignment(cost)
    best = sum(cost[i][j] for i, j in optimum)
    for r, c in optimum:
        blocked = [list(row) for row in cost]
        blocked[r][c] = float("inf")
        try:
            alternative = hungarian_assignment(blocked)
        except ValueError:
            continue
        if sum(blocked[i][j] for i, j in alternative) <= best + tolerance:
            return False

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for _ in range(min(len(cost), len(cost[0]))):
        available = [
            cost[i][j] for i in range(len(cost)) if i not in used_rows
            for j in range(len(cost[0])) if j not in used_cols
        ]
        cheapest = min(available)
        if sum(value <= cheapest + tolerance for value in available) > 1:
            return False
        for i in range(len(cost)):
            for j in range(len(cost[0])):
                if i not in used_rows and j not in used_cols and cost[i][j] == cheapest:
                    used_rows.add(i)
                    used_cols.add(j)
                    break
            else:
                continue
            break
    return True


def score(
    cost: Sequence[Sequence[float]],
    true_pairs: Sequence[tuple[int, int]],
    row_order: Sequence[int],
    col_order: Sequence[int],
) -> dict:
    view = [[cost[i][j] for j in col_order] for i in row_order]
    truth = set(true_pairs)
    recalls = [
        len(truth & {(row_order[i], col_order[j]) for i, j in assign(view)}) / len(truth)
        for assign in (hungarian_assignment, greedy_assignment)
    ]
    return {"n_true": len(truth), "recall_hungarian": recalls[0], "recall_greedy": recalls[1]}


def load_true_pairs(path: Path) -> dict[tuple, list[tuple[int, int]]]:
    csv.field_size_limit(10 ** 7)
    truths = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("true_assignment"):
                key = tuple(row[field] for field in MATRIX_KEY)
                truths[key] = [(int(i), int(j)) for i, j in json.loads(row["true_assignment"])]
    return truths


def load_matrices(
    costs_path: Path, truths: dict[tuple, list[tuple[int, int]]], measures: Sequence[str] | None = None
) -> Iterator[tuple[tuple, list[list[float]], list[tuple[int, int]]]]:
    wanted = None if measures is None else set(measures)
    with gzip.open(costs_path, "rt") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except (ValueError, EOFError):
                break
            cost = record.get("cost")
            if (wanted is not None and record["measure"] not in wanted) or not cost or not cost[0]:
                continue
            key = tuple(str(record[field]) for field in MATRIX_KEY)
            if truths.get(key):
                yield key, cost, truths[key]


def run(
    results: Path,
    permutations: int,
    seed: int = 0,
    datasets: Sequence[str] | None = None,
    measures: Sequence[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    paths = [
        path for path in sorted(results.glob("costs_synthetic_*_set_random_uniform.jsonl.gz"))
        if (datasets is None or path.name[len("costs_synthetic_"):].split("_")[0] in datasets)
        and (results / f"predictions_{path.name[len('costs_'):-len('.jsonl.gz')]}.csv").exists()
    ]
    for costs_path in tqdm(paths, desc="runs", unit="run"):
        stem = costs_path.name[len("costs_"):-len(".jsonl.gz")]
        truths = load_true_pairs(results / f"predictions_{stem}.csv")
        rng = random.Random(seed)
        for key, cost, true_pairs in tqdm(
            load_matrices(costs_path, truths, measures), desc=f"  {stem}", leave=False, unit="matrix"
        ):
            order = (list(range(len(cost))), list(range(len(cost[0]))))
            decisive = is_decisive(cost)
            for index in range(permutations):
                rows.append(dict(zip(MATRIX_KEY, key)) | {"permutation": index, "decisive": decisive}
                            | score(cost, true_pairs, *order))
                order = (random.Random(rng.random()).sample(range(len(cost)), len(cost)),
                         random.Random(rng.random()).sample(range(len(cost[0])), len(cost[0])))
    return rows


def summarise(rows: Sequence[dict]) -> list[dict]:
    by_measure: dict[str, list[dict]] = {}
    for row in rows:
        by_measure.setdefault(row["measure"], []).append(row)

    out = []
    for measure, group in sorted(by_measure.items()):
        seen: dict[tuple, set[tuple[float, float]]] = {}
        for row in group:
            seen.setdefault(tuple(row[f] for f in MATRIX_KEY), set()).add(
                (row["recall_hungarian"], row["recall_greedy"])
            )
        def weighted(subset, side):
            n = sum(r["n_true"] for r in subset)
            return sum(r[f"recall_{side}"] * r["n_true"] for r in subset) / n if n else float("nan")

        firm = [r for r in group if r["decisive"]]
        out.append({
            "measure": measure, "matrices": len(seen),
            "recall_hungarian": weighted(group, "hungarian"),
            "recall_greedy": weighted(group, "greedy"),
            "recall_gap": weighted(group, "hungarian") - weighted(group, "greedy"),
            "unstable": sum(len(v) > 1 for v in seen.values()) / len(seen),
            "decisive": len({tuple(r[f] for f in MATRIX_KEY) for r in firm}) / len(seen),
            "recall_hungarian_decisive": weighted(firm, "hungarian"),
            "recall_greedy_decisive": weighted(firm, "greedy"),
            "recall_gap_decisive": weighted(firm, "hungarian") - weighted(firm, "greedy"),
        })
    return out


def format_summary(summary: Sequence[dict]) -> str:
    header = (f"{'measure':<38}{'matrices':>10}{'rec H':>9}{'rec G':>9}{'H-G':>10}{'unstable':>10}"
              f"{'decisive':>10}{'rec H|d':>9}{'rec G|d':>9}{'H-G|d':>10}")
    lines = [header, "-" * len(header)]
    for row in summary:
        lines.append(
            f"{row['measure']:<38}{row['matrices']:>10}{row['recall_hungarian']:>9.4f}"
            f"{row['recall_greedy']:>9.4f}{row['recall_gap']:>+10.4f}{row['unstable']:>10.3f}"
            f"{row['decisive']:>10.3f}{row['recall_hungarian_decisive']:>9.4f}"
            f"{row['recall_greedy_decisive']:>9.4f}{row['recall_gap_decisive']:>+10.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Default: every dataset found.")
    parser.add_argument("--permutations", type=int, default=30,
                        help="Layouts scored per matrix; the first is always the run's own.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measures", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/matching_ablation.csv"))
    args = parser.parse_args()

    rows = run(args.results, args.permutations, args.seed, args.datasets, args.measures)
    if not rows:
        print("no matrices produced")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} scored layouts -> {args.out}\n")
    print(format_summary(summarise(rows)))


if __name__ == "__main__":
    main()
