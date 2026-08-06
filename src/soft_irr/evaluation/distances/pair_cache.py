
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

CACHEABLE_MEASURES = frozenset({
    "nli_mednli",
    "nli_mdeberta",
    "logprob_qwen35",
    "logprob_medgemma",
    "verbalized_qwen35",
    "verbalized_medgemma",
})

_JOIN_KEY = ("agreement_level", "seed", "measure", "sample_index")


def canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def base_measure(name: str) -> str:
    return name.removesuffix("_single")


_TORN = (EOFError, OSError, ValueError, KeyError, TypeError)


class PairCostWriter:

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._fh = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(self.path, "at")
        self.n_written = 0

    def write(self, measure: str, reasoning: bool, items: Mapping[tuple[str, str], float]) -> None:
        if self._fh is None or not items:
            return
        for (x, y), cost in items.items():
            self._fh.write(json.dumps(
                {"measure": measure, "reasoning": reasoning, "x": x, "y": y, "cost": round(float(cost), 6)},
                ensure_ascii=False,
            ) + "\n")
        self.n_written += len(items)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "PairCostWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def attach_writer(measures: Mapping[str, Any], writer: PairCostWriter) -> int:
    seen: set[int] = set()
    attached = 0
    for name, measure in measures.items():
        target = getattr(measure, "_base", measure)
        if id(target) in seen or not getattr(target, "is_pair_cacheable", False):
            continue
        seen.add(id(target))
        measure_name = base_measure(name)
        target._pair_cost_sink = (
            lambda items, m=measure_name, t=target: writer.write(m, t.enable_reasoning, items)
        )
        attached += 1
    return attached


def _load_predictions(path: Path, reasoning: bool) -> dict[tuple[str, ...], tuple[list[str], list[str]]]:
    csv.field_size_limit(10 ** 7)
    rows: dict[tuple[str, ...], tuple[list[str], list[str]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except _TORN:
                break
            try:
                if (row["reasoning"] == "True") != reasoning:
                    continue
                rows[tuple(row[k] for k in _JOIN_KEY)] = (
                    json.loads(row["rater_a"]), json.loads(row["rater_b"])
                )
            except _TORN:
                break
    return rows


def _iter_cost_records(path: Path):
    with gzip.open(path, "rt") as f:
        while True:
            try:
                line = next(f)
                record = json.loads(line)
            except StopIteration:
                return
            except _TORN:
                return
            yield record


def _read_pair_costs(
    path: Path, keep: set[str], reasoning: bool, costs: dict[str, dict[tuple[str, str], float]]
) -> None:
    for record in _iter_cost_records(path):
        measure = base_measure(record["measure"])
        if measure not in keep or bool(record["reasoning"]) != reasoning:
            continue
        costs.setdefault(measure, {})[canonical(record["x"], record["y"])] = record["cost"]


def wanted_measures(measures: Iterable[str]) -> set[str]:
    return {base_measure(name) for name in measures} & CACHEABLE_MEASURES


def load_costs(
    results_dir: str | Path,
    wanted: Iterable[str] | None = None,
    reasoning: bool = False,
    stats: dict[str, int] | None = None,
) -> dict[str, dict[tuple[str, str], float]]:
    results_dir = Path(results_dir)
    keep = CACHEABLE_MEASURES if wanted is None else (set(wanted) & CACHEABLE_MEASURES)
    counts: dict[str, int] = {
        "files": 0, "unmatched": 0, "shape_mismatch": 0, "missing_predictions": 0, "unreadable": 0,
    }
    costs: dict[str, dict[tuple[str, str], float]] = {}
    if not keep:
        if stats is not None:
            stats.update(counts)
        return costs
    paths = [*results_dir.glob("costs_*.jsonl.gz"), *results_dir.glob("paircosts_*.jsonl.gz")]
    for cost_path in sorted(paths, key=lambda p: p.stat().st_mtime):
        counts["files"] += 1
        if cost_path.name.startswith("paircosts_"):
            _read_pair_costs(cost_path, keep, reasoning, costs)
            continue
        prediction_path = results_dir / f"predictions_{cost_path.name[len('costs_'):-len('.jsonl.gz')]}.csv"
        if not prediction_path.exists():
            counts["missing_predictions"] += 1
            continue
        try:
            predictions = _load_predictions(prediction_path, reasoning)
        except _TORN:
            counts["unreadable"] += 1
            continue
        for record in _iter_cost_records(cost_path):
            measure = base_measure(record["measure"])
            if measure not in keep:
                continue
            entry = predictions.get(tuple(str(record[k]) for k in _JOIN_KEY))
            if entry is None:
                counts["unmatched"] += 1
                continue
            rater_a, rater_b = entry
            matrix = record["cost"]
            if len(matrix) != len(rater_a) or any(len(row) != len(rater_b) for row in matrix):
                counts["shape_mismatch"] += 1
                continue
            entries = costs.setdefault(measure, {})
            for i, x in enumerate(rater_a):
                for j, y in enumerate(rater_b):
                    entries[canonical(x, y)] = matrix[i][j]
    if stats is not None:
        stats.update(counts)
    return costs


def attach(measures: Mapping[str, Any], costs: Mapping[str, dict[tuple[str, str], float]]) -> int:
    seen: set[int] = set()
    attached = 0
    for name, measure in measures.items():
        entries = costs.get(base_measure(name))
        target = getattr(measure, "_base", measure)
        if not entries or id(target) in seen or not getattr(target, "is_pair_cacheable", False):
            continue
        seen.add(id(target))
        target._pair_cache = entries
        attached += 1
    return attached
