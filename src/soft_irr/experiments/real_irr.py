
from __future__ import annotations

import argparse
import csv
import json
import timeit
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from soft_irr.common import load_dotenv
from soft_irr.evaluation.distances import DistanceFunction, set_jaccard_distance
from soft_irr.evaluation.distances.base import matching_strategies
from soft_irr.evaluation.distances.pair_cache import (
    PairCostWriter,
    attach,
    attach_writer,
    load_costs,
    wanted_measures,
)
from soft_irr.evaluation.scores import AgreementScores, MultiRaterAgreementScores
from soft_irr.experiments.load_annotations import (
    load_derm1_annotation_rows,
    load_passion_annotation_rows,
    load_reflacx_annotation_rows,
)
from soft_irr.experiments.synthetic_irr import _build_measures, _mae_mape, _needs_prebatch, _summary_key

_ROOT_DIR = Path(__file__).parents[3]
load_dotenv(str(_ROOT_DIR / ".env"))

_RESULT_FIELDS = [
    "dataset", "category_field", "measure", "matching", "n_items",
    "true_avg_agreement", "true_cohens_kappa", "true_fleiss_kappa", "true_sigma", "true_ks",
    "avg_agreement", "cohens_kappa", "fleiss_kappa", "sigma", "ks",
    "mae_avg_agreement", "mape_avg_agreement", "mae_kappa", "mape_kappa",
    "mae_fleiss_kappa", "mape_fleiss_kappa", "mae_sigma", "mape_sigma", "mae_ks", "mape_ks",
]

_PREDICTION_FIELDS = [
    "dataset", "category_field", "measure", "phash", "rater_a_idx", "rater_b_idx",
    "caption_a", "caption_b", "category_a", "category_b", "distance",
]

_TIMING_FIELDS = ["dataset", "category_field", "measure", "n_pairs", "elapsed_seconds", "pairs_per_second"]

_CHANCE_FIELDS = ["dataset", "category_field", "measure", "matching", "phash", "weight", "distance"]


def _write_chance_sample(
    writer: csv.DictWriter, dataset: str, category_field: str, measure: str,
    keys: list[str], captions: list[list[list[str]]], dist_fn: DistanceFunction,
) -> None:
    owner: dict[frozenset[str], str] = {}
    pooled_labels: list[frozenset[str]] = []
    for key, item in zip(keys, captions):
        for rater in item:
            labels = frozenset(rater)
            pooled_labels.append(labels)
            owner.setdefault(labels, key)

    pairs, weights = AgreementScores.chance_pairs_and_weights(pooled_labels, pooled_labels)
    strategies = matching_strategies(dist_fn)
    unique_pairs = list(dict.fromkeys(pairs))
    distances = AgreementScores._batch_by_strategy(dist_fn, unique_pairs, strategies)
    for matching in strategies:
        cache = dict(zip(unique_pairs, distances[matching]))
        for pair, weight in zip(pairs, weights):
            writer.writerow({
                "dataset": dataset, "category_field": category_field, "measure": measure,
                "matching": matching, "phash": owner[pair[0]], "weight": weight,
                "distance": cache[pair],
            })


def _build_items(
    rows_by_hash: dict[str, list[tuple[str, frozenset[str], str]]],
) -> tuple[list[str], list[list[list[str]]], list[list[list[str]]]]:
    keys: list[str] = []
    captions: list[list[list[str]]] = []
    categories: list[list[list[str]]] = []
    for key, rows in rows_by_hash.items():
        seen_sources: set[str] = set()
        item_captions: list[list[str]] = []
        item_categories: list[list[str]] = []
        for caption, category, source in rows:
            if source in seen_sources:
                continue
            seen_sources.add(source)
            item_captions.append([caption])
            item_categories.append(list(category))
        if len(item_captions) >= 2:
            keys.append(key)
            captions.append(item_captions)
            categories.append(item_categories)
    return keys, captions, categories


def _prebatch_items(
    dist_fn: DistanceFunction, captions: list[list[list[str]]], name: str, batch_size: int
) -> dict | None:
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    pairs: list[tuple[list[str], list[str]]] = []

    def _queue(a: list[str], b: list[str]) -> None:
        key = (tuple(a), tuple(b))
        if key not in seen:
            seen.add(key)
            pairs.append((list(a), list(b)))

    by_slot_pair: dict[tuple[int, int], list[tuple[list[str], list[str]]]] = defaultdict(list)
    pooled: list[list[str]] = []
    for item in captions:
        pooled.extend(item)
        for s in range(len(item)):
            for t in range(s + 1, len(item)):
                _queue(item[s], item[t])
                _queue(item[t], item[s])
                by_slot_pair[(s, t)].append((item[s], item[t]))

    pooled_labels = [frozenset(c) for c in pooled]
    chance_pairs, _ = AgreementScores.chance_pairs_and_weights(pooled_labels, pooled_labels)
    for a, b in chance_pairs:
        _queue(a, b)
    for slot_pairs in by_slot_pair.values():
        a_labels = [frozenset(a) for a, _ in slot_pairs]
        b_labels = [frozenset(b) for _, b in slot_pairs]
        chance_pairs, _ = AgreementScores.chance_pairs_and_weights(a_labels, b_labels)
        for a, b in chance_pairs:
            _queue(a, b)

    if not pairs:
        return None
    tqdm.write(f"  Pre-computing {name}: {len(pairs):,} unique label pairs (batch_size={batch_size}) ...")
    start = timeit.default_timer()
    dist_fn.batch(pairs, batch_size=batch_size)
    elapsed = timeit.default_timer() - start
    pairs_per_second = len(pairs) / elapsed
    tqdm.write(f"  Pre-computing {name}: done in {elapsed:.2f}s ({pairs_per_second:,.1f} pairs/s)")
    return {"measure": name, "n_pairs": len(pairs), "elapsed_seconds": elapsed, "pairs_per_second": pairs_per_second}


def run_experiment(
    rows_by_hash: dict[str, list[tuple[str, frozenset[str], str]]],
    category_field: str,
    measures: dict[str, DistanceFunction],
    dataset: str,
    predictions_writer: csv.DictWriter | None = None,
    chance_writer: csv.DictWriter | None = None,
    batch_size: int = 32,
) -> tuple[list[dict], list[dict]]:
    measures = {name: fn for name, fn in measures.items() if not fn.is_bipartite}

    keys, captions, categories = _build_items(rows_by_hash)
    print(f"  {len(keys):,} images with >=2 independent raters after de-duplicating same-source rows.")
    true_scores = MultiRaterAgreementScores.get(categories, distance_function=set_jaccard_distance)

    rows: list[dict] = []
    timing_rows: list[dict] = []
    for name, dist_fn in tqdm(measures.items(), desc="measures", unit="measure"):
        if _needs_prebatch(dist_fn):
            timing = _prebatch_items(dist_fn, captions, name, batch_size)
            if timing is not None:
                timing_rows.append({"dataset": dataset, "category_field": category_field, **timing})

        try:
            scores_by_matching = MultiRaterAgreementScores.get_by_matching(
                captions, distance_function=dist_fn, chance_distance_function=dist_fn,
            )
            for matching, scores in scores_by_matching.items():
                mae_avg_agreement, mape_avg_agreement = _mae_mape(
                    scores.average_agreement, true_scores.average_agreement
                )
                mae_kappa, mape_kappa = _mae_mape(scores.cohens_kappa, true_scores.cohens_kappa)
                mae_fleiss_kappa, mape_fleiss_kappa = _mae_mape(scores.fleiss_kappa, true_scores.fleiss_kappa)
                mae_sigma, mape_sigma = _mae_mape(scores.sigma, true_scores.sigma)
                mae_ks, mape_ks = _mae_mape(scores.ks, true_scores.ks)
                rows.append({
                    "dataset": dataset,
                    "category_field": category_field,
                    "measure": name,
                    "matching": matching,
                    "n_items": len(keys),
                    "true_avg_agreement": true_scores.average_agreement,
                    "true_cohens_kappa": true_scores.cohens_kappa,
                    "true_fleiss_kappa": true_scores.fleiss_kappa,
                    "true_sigma": true_scores.sigma,
                    "true_ks": true_scores.ks,
                    "avg_agreement": scores.average_agreement,
                    "cohens_kappa": scores.cohens_kappa,
                    "fleiss_kappa": scores.fleiss_kappa,
                    "sigma": scores.sigma,
                    "ks": scores.ks,
                    "mae_avg_agreement": mae_avg_agreement,
                    "mape_avg_agreement": mape_avg_agreement,
                    "mae_kappa": mae_kappa,
                    "mape_kappa": mape_kappa,
                    "mae_fleiss_kappa": mae_fleiss_kappa,
                    "mape_fleiss_kappa": mape_fleiss_kappa,
                    "mae_sigma": mae_sigma,
                    "mape_sigma": mape_sigma,
                    "mae_ks": mae_ks,
                    "mape_ks": mape_ks,
                })
            if predictions_writer is not None:
                all_pairs: list[tuple[list[str], list[str]]] = []
                meta: list[tuple[str, int, int, list[str], list[str]]] = []
                for key, item_captions, item_categories in zip(keys, captions, categories):
                    n = len(item_captions)
                    for s in range(n):
                        for t in range(s + 1, n):
                            all_pairs.append((item_captions[s], item_captions[t]))
                            meta.append((key, s, t, item_categories[s], item_categories[t]))
                sample_dists = dist_fn.batch(all_pairs, batch_size=batch_size)
                for (ca, cb), (key, s, t, cat_a, cat_b), dist in zip(all_pairs, meta, sample_dists):
                    predictions_writer.writerow({
                        "dataset": dataset,
                        "category_field": category_field,
                        "measure": name,
                        "phash": key,
                        "rater_a_idx": s,
                        "rater_b_idx": t,
                        "caption_a": ca[0],
                        "caption_b": cb[0],
                        "category_a": json.dumps(cat_a, ensure_ascii=False),
                        "category_b": json.dumps(cat_b, ensure_ascii=False),
                        "distance": dist,
                    })
            if chance_writer is not None:
                _write_chance_sample(chance_writer, dataset, category_field, name, keys, captions, dist_fn)
        except Exception as exc:
            tqdm.write(f"  [WARN] {name} failed: {exc}")
        dist_fn.release()

    return rows, timing_rows


def _print_summary(rows: list[dict]) -> None:
    print("\n  Results by measure (single deterministic pass over all images):")
    header = (
        f"    {'measure':<30s}  {'avg_agreement MAE':>18s}  {'cohens_kappa MAE':>18s}  "
        f"{'fleiss_kappa MAE':>18s}  {'sigma MAE':>10s}  {'ks MAE':>10s}"
    )
    print(header)
    for r in sorted(rows, key=lambda r: r["mae_avg_agreement"]):
        print(
            f"    {_summary_key(r):<30s}  {r['mae_avg_agreement']:>18.4f}  {r['mae_kappa']:>18.4f}  "
            f"{r['mae_fleiss_kappa']:>18.4f}  {r['mae_sigma']:>10.4f}  {r['mae_ks']:>10.4f}"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real IRR on Derm1M or REFLACX annotation clusters.")
    p.add_argument(
        "--dataset", default="derm1", choices=["derm1", "reflacx", "passion"],
        help=(
            "'derm1': duplicate-image caption clusters. 'reflacx': chest x-ray transcriptions "
            "from phases 1+2. 'passion': independently-written lesion descriptions."
        ),
    )
    p.add_argument(
        "--category-field", default=None,
        choices=["disease_label", "skin_concept", "both", "diagnosis", "conditions_PASSION"],
        help=(
            "(required for derm1 and passion) Which mapped category is the ground-truth label. "
            "derm1: 'disease_label' (one leaf diagnosis), 'skin_concept' (set of findings), or "
            "'both' (their union, namespaced by field). passion: 'diagnosis' (45-class) or "
            "'conditions_PASSION' (4-class)."
        ),
    )
    p.add_argument("--cache-dir", default=None, help="(derm1 only) Optional HuggingFace datasets cache directory.")
    p.add_argument("--hash-cache-path", default=None, help="(derm1 only) Perceptual-hash cache CSV (default: data/derm1_image_hashes.csv).")
    p.add_argument("--archive-tmp-dir", default=None, help="(derm1 only) Scratch dir for one-archive-at-a-time downloads.")
    p.add_argument("--reflacx-dir", default=None, help="(reflacx only) Root directory of the REFLACX dataset (default: data/reflacx).")
    p.add_argument("--passion-dir", default=None, help="(passion only) Root directory of the PASSION dataset (default: data/passion).")
    p.add_argument("--max-rows", type=int, default=None, help="Limit metadata rows scanned (for a quick test run).")
    p.add_argument("--output", default="results/", help="Directory for output CSV files.")
    p.add_argument("--include-llm", action="store_true", help="Also run slow LLM-based measures. Requires GPU or API key.")
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for model-backed measures' GPU inference calls. Lower this if you hit CUDA OOM.",
    )
    p.add_argument("--simple-only", action="store_true", help="Only calculate simple string distances.")
    p.add_argument(
        "--no-cache", dest="use_pair_cache", action="store_false",
        help=(
            "Re-run all LLM/NLI inference instead of reusing the edge costs recorded in earlier "
            "synthetic runs' costs_*.jsonl.gz sidecars."
        ),
    )
    p.add_argument(
        "--cache-from", default=None,
        help="Directory of earlier runs to recover edge costs from (default: --output).",
    )
    args = p.parse_args()
    if args.dataset in ("derm1", "passion") and args.category_field is None:
        p.error(f"--category-field is required for --dataset {args.dataset}")
    if args.dataset == "derm1" and args.category_field not in ("disease_label", "skin_concept", "both"):
        p.error("--category-field must be 'disease_label', 'skin_concept' or 'both' for --dataset derm1")
    if args.dataset == "passion" and args.category_field not in ("diagnosis", "conditions_PASSION"):
        p.error("--category-field must be 'diagnosis' or 'conditions_PASSION' for --dataset passion")
    return args


def main() -> None:
    args = _parse_args()
    dataset = args.dataset

    if dataset == "derm1":
        category_field = args.category_field
        print(f"Loading Derm1M annotation rows (category_field={category_field}) ...")
        rows_by_hash = load_derm1_annotation_rows(
            category_field,
            cache_dir=args.cache_dir,
            hash_cache_path=args.hash_cache_path,
            max_rows=args.max_rows,
            archive_tmp_dir=args.archive_tmp_dir,
        )
        print(f"  {len(rows_by_hash):,} images with >=2 independently-captioned rows and a valid {category_field}.")
    elif dataset == "reflacx":
        category_field = "findings"
        print("Loading REFLACX transcription rows (phases 1+2) ...")
        rows_by_hash = load_reflacx_annotation_rows(reflacx_dir=args.reflacx_dir, max_rows=args.max_rows)
        print(f"  {len(rows_by_hash):,} chest x-rays with >=2 independently-transcribed readings.")
    else:
        category_field = args.category_field
        print(f"Loading PASSION description rows (category_field={category_field}) ...")
        rows_by_hash = load_passion_annotation_rows(
            category_field, passion_dir=args.passion_dir, max_rows=args.max_rows,
        )
        print(f"  {len(rows_by_hash):,} images with >=2 independently-written descriptions and a valid {category_field}.")

    if not rows_by_hash:
        print("  [ERROR] No qualifying rows found.")
        return

    measures = _build_measures(args.include_llm, args.simple_only)
    if args.use_pair_cache and wanted_measures(measures):
        cache_dir = args.cache_from or args.output
        stats: dict[str, int] = {}
        costs = load_costs(cache_dir, wanted_measures(measures), stats=stats)
        attached = attach(measures, costs)
        print(
            f"Edge-cost cache: {sum(len(v) for v in costs.values()):,} pairs recovered from "
            f"{stats['files']:,} runs in {cache_dir}, backing {attached} measures"
        )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / f"predictions_real_{dataset}_{category_field}.csv"
    chance_path = out_dir / f"chance_real_{dataset}_{category_field}.csv"
    pair_cost_path = out_dir / f"paircosts_real_{dataset}_{category_field}.jsonl.gz"
    with open(predictions_path, "w", newline="", encoding="utf-8") as pred_f, open(
        chance_path, "w", newline="", encoding="utf-8"
    ) as chance_f, PairCostWriter(pair_cost_path) as pair_cost_writer:
        attach_writer(measures, pair_cost_writer)
        predictions_writer = csv.DictWriter(pred_f, fieldnames=_PREDICTION_FIELDS)
        predictions_writer.writeheader()
        chance_writer = csv.DictWriter(chance_f, fieldnames=_CHANCE_FIELDS)
        chance_writer.writeheader()

        rows, timing_rows = run_experiment(
            rows_by_hash=rows_by_hash,
            category_field=category_field,
            measures=measures,
            dataset=dataset,
            predictions_writer=predictions_writer,
            chance_writer=chance_writer,
            batch_size=args.batch_size,
        )
    print(f"  Saved sample-level predictions -> {predictions_path}")
    print(f"  Saved the pooled chance sample -> {chance_path}")

    timing_path = out_dir / f"timing_real_{dataset}_{category_field}.csv"
    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(timing_rows)
    print(f"  Saved {len(timing_rows):,} precomputation timing rows -> {timing_path}")

    out_path = out_dir / f"real_{dataset}_{category_field}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows -> {out_path}")

    _print_summary(rows)


if __name__ == "__main__":
    main()
