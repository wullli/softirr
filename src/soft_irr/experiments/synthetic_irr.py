
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import timeit
from collections import defaultdict
from pathlib import Path

import numpy as np
from mlxtend.evaluate import permutation_test
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from soft_irr.common import load_dotenv
from soft_irr.experiments.load_taxonomies import (
    ConceptEntry,
    build_hierarchy_index,
    build_lexical_index,
    build_sibling_index,
    load_icd11,
    load_icd11_ancestor_hierarchy,
    load_icd11_ancestor_titles,
    load_meddra,
    load_meddra_ancestor_hierarchy,
    load_meddra_ancestor_names,
    load_mesh,
)
from soft_irr.evaluation.distances import (
    BipartiteMatchingDistance,
    DistanceFunction,
    bipartite_embedding_distance,
    bipartite_jaccard_distance,
    bipartite_lcsseq_distance,
    bipartite_levenshtein_distance,
    bipartite_nli_distance,
    bipartite_nli_multilingual_distance,
    bipartite_normalized_indel_distance,
    bipartite_qwen3_embedding_distance,
    exact_match_distance,
    jaccard_distance,
    lcsseq_distance,
    levenshtein_distance,
    masi_distance,
    normalized_indel_similarity,
    set_jaccard_distance,
    single_embedding_pubmedbert,
    single_embedding_qwen3,
    single_nli_mednli,
    single_nli_multilingual,
)
from soft_irr.evaluation.distances.base import _as_matching_str_list
from soft_irr.evaluation.distances.pair_cache import (
    PairCostWriter,
    attach,
    attach_writer,
    load_costs,
    wanted_measures,
)
from soft_irr.evaluation.scores import (
    DEFAULT_MAX_EXACT_CHANCE_PAIRS,
    DEFAULT_MC_MIN_SHUFFLES,
    DEFAULT_MC_TARGET_SAMPLES,
    AgreementScores,
)

_ROOT_DIR = Path(__file__).parents[3]
load_dotenv(str(_ROOT_DIR / ".env"))


def _negative_candidates(
    anchor: ConceptEntry,
    concepts: list[ConceptEntry],
    sibling_index: dict[str, list[str]] | None,
    concept_by_code: dict[str, ConceptEntry] | None,
    negative_index: dict[str, list[ConceptEntry]] | None,
) -> list[ConceptEntry]:
    if negative_index is not None:
        return negative_index[anchor.code]
    if sibling_index is not None and concept_by_code is not None:
        return [concept_by_code[code] for code in sibling_index[anchor.code]]
    return concepts


def _draw_negative(
    rng: random.Random,
    anchor: ConceptEntry,
    candidates: list[ConceptEntry],
    used_codes: set[str],
    max_tries: int = 20,
) -> ConceptEntry:
    for _ in range(max_tries):
        candidate = rng.choice(candidates)
        if candidate.code not in used_codes:
            return candidate
    pool = [c for c in candidates if c.code not in used_codes]
    if pool:
        return rng.choice(pool)
    return rng.choice([c for c in candidates if c.code != anchor.code] or candidates)


def _generate_single_pairs(
    concepts: list[ConceptEntry],
    n: int,
    agreement_level: float,
    rng: random.Random,
    sibling_index: dict[str, list[str]] | None = None,
    concept_by_code: dict[str, ConceptEntry] | None = None,
    negative_index: dict[str, list[ConceptEntry]] | None = None,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
    if not concepts:
        raise ValueError("Concept pool is empty; nothing to sample single-label pairs from.")
    rater_a: list[list[str]] = []
    rater_b: list[list[str]] = []
    rater_a_concepts: list[list[str]] = []
    rater_b_concepts: list[list[str]] = []
    for _ in range(n):
        if rng.random() < agreement_level:
            c = rng.choice(concepts)
            rater_a.append([rng.choice(c.synonyms)])
            rater_a_concepts.append([c.preferred_term])
            rater_b.append([rng.choice(c.synonyms)])
            rater_b_concepts.append([c.preferred_term])
        else:
            c1 = rng.choice(concepts)
            rater_a.append([rng.choice(c1.synonyms)])
            rater_a_concepts.append([c1.preferred_term])
            candidates = _negative_candidates(
                c1, concepts, sibling_index, concept_by_code, negative_index
            )
            c2 = _draw_negative(rng, c1, candidates, {c1.code})
            rater_b.append([rng.choice(c2.synonyms)])
            rater_b_concepts.append([c2.preferred_term])
    return rater_a, rater_b, rater_a_concepts, rater_b_concepts


def _n_shared(target_jaccard: float, a_size: int, b_size: int, rng: random.Random) -> int:
    cap = min(a_size, b_size)
    k = target_jaccard * (a_size + b_size) / (1.0 + target_jaccard)
    k_lo = min(math.floor(k), cap)
    k_hi = min(k_lo + 1, cap)
    if k_hi == k_lo:
        return k_lo
    j_lo = k_lo / (a_size + b_size - k_lo)
    j_hi = k_hi / (a_size + b_size - k_hi)
    p_hi = min(max((target_jaccard - j_lo) / (j_hi - j_lo), 0.0), 1.0)
    return k_hi if rng.random() < p_hi else k_lo


def _generate_set_pairs(
    concepts: list[ConceptEntry],
    n: int,
    agreement_level: float,
    rng: random.Random,
    sibling_index: dict[str, list[str]] | None = None,
    concept_by_code: dict[str, ConceptEntry] | None = None,
    negative_index: dict[str, list[ConceptEntry]] | None = None,
    min_set_size: int = 1,
    max_set_size: int = 10,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
    if len(concepts) < min_set_size:
        raise ValueError(
            f"Concept pool has {len(concepts)} concept(s), fewer than min_set_size={min_set_size}."
        )
    rater_a: list[list[str]] = []
    rater_b: list[list[str]] = []
    rater_a_concepts: list[list[str]] = []
    rater_b_concepts: list[list[str]] = []
    max_gap = max_set_size - min_set_size
    gap = round((1.0 - agreement_level) * max_gap)

    for _ in range(n):
        a_size = min(rng.randint(min_set_size, max_set_size), len(concepts))
        item_gap = min(gap, math.floor(a_size * (1.0 - agreement_level)))
        if item_gap > 0:
            b_size = max(
                min_set_size, min(max_set_size, a_size + rng.randint(-item_gap, item_gap))
            )
        else:
            b_size = a_size
        n_agree = _n_shared(agreement_level, a_size, b_size, rng)

        shared = rng.sample(concepts, a_size)
        rater_a_set = [rng.choice(c.synonyms) for c in shared]
        rater_a_concept_set = [c.preferred_term for c in shared]

        rater_b_set: list[str] = []
        rater_b_concept_set: list[str] = []
        for concept in shared[:n_agree]:
            rater_b_set.append(rng.choice(concept.synonyms))
            rater_b_concept_set.append(concept.preferred_term)

        leftover_a = shared[n_agree:]
        used_codes = {c.code for c in shared}
        for k in range(b_size - n_agree):
            anchor = leftover_a[k] if k < len(leftover_a) else rng.choice(shared)
            candidates = _negative_candidates(
                anchor, concepts, sibling_index, concept_by_code, negative_index
            )
            other = _draw_negative(rng, anchor, candidates, used_codes)
            used_codes.add(other.code)
            rater_b_set.append(rng.choice(other.synonyms))
            rater_b_concept_set.append(other.preferred_term)

        rater_a.append(rater_a_set)
        rater_b.append(rater_b_set)
        rater_a_concepts.append(rater_a_concept_set)
        rater_b_concepts.append(rater_b_concept_set)

    return rater_a, rater_b, rater_a_concepts, rater_b_concepts


_RESULT_FIELDS = [
    "dataset", "mode", "negative_mode", "agreement_level", "seed",
    "measure", "matching", "chance_measure", "reasoning",
    "true_avg_agreement", "true_cohens_kappa", "true_scotts_pi", "true_sigma", "true_ks",
    "avg_agreement", "cohens_kappa", "scotts_pi", "sigma", "ks",
    "mae_avg_agreement", "mape_avg_agreement", "mae_kappa", "mape_kappa", "mae_pi", "mape_pi",
    "mae_sigma", "mape_sigma", "mae_ks", "mape_ks",
]

_PREDICTION_FIELDS = [
    "dataset", "mode", "negative_mode", "agreement_level", "seed",
    "measure", "chance_measure", "reasoning", "sample_index",
    "rater_a_set_size", "rater_b_set_size", "rater_a", "rater_b", "distance",
    "distance_oracle", "distance_greedy", "true_match", "true_assignment", "predicted_assignment",
    "greedy_assignment", "confident_assignment",
]

_DRAFT_MAX_EXACT_CHANCE_PAIRS = 50

_COST_MATRIX_FIELDS = [
    "dataset", "mode", "negative_mode", "agreement_level", "seed", "measure", "sample_index", "cost",
]

_Population = tuple[
    float, int, list[list[str]], list[list[str]], list[list[str]], list[list[str]], AgreementScores,
]


class CostMatrixWriter:

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._fh = gzip.open(path, "wt", encoding="utf-8") if path is not None else None
        self.n_written = 0

    def write(self, key: dict, cost: list[list[float]]) -> None:
        if self._fh is None:
            return
        record = {**key, "cost": [[round(float(c), 6) for c in row] for row in cost]}
        self._fh.write(json.dumps({k: record[k] for k in _COST_MATRIX_FIELDS}) + "\n")
        self.n_written += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()

    def __enter__(self) -> CostMatrixWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def output_suffix(args: argparse.Namespace, mode: str) -> str:
    suffix = "_uniform" if mode == "set" else ""
    suffix += "_draft" if args.draft and not args.chance_measure else (
        f"_chance-{args.chance_measure}" if args.chance_measure else ""
    )
    return suffix + ("_reasoning" if args.enable_reasoning else "")


def chance_settings(args: argparse.Namespace) -> tuple[str | None, int | None]:
    max_exact_chance_pairs = args.max_exact_chance_pairs
    if max_exact_chance_pairs is None and args.draft:
        max_exact_chance_pairs = _DRAFT_MAX_EXACT_CHANCE_PAIRS
    return args.chance_measure, max_exact_chance_pairs


def load_cost_matrices(path: str | Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mae_mape(value: float, truth: float) -> tuple[float, float]:
    mae = abs(value - truth)
    return mae, (mae / abs(truth) * 100.0) if truth != 0 else float("nan")


def _true_assignment(a_concepts: list[str], b_concepts: list[str]) -> list[tuple[int, int]]:
    if not a_concepts or not b_concepts:
        return []
    cost = [[0.0 if a == b else 1.0 for b in b_concepts] for a in a_concepts]
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r][c] == 0.0]


def _generate_populations(
    mode: str,
    pool: list[ConceptEntry],
    n_samples: int,
    seeds: list[int],
    sibling_index: dict[str, list[str]] | None,
    concept_by_code: dict[str, ConceptEntry],
    negative_index: dict[str, list[ConceptEntry]] | None = None,
    min_set_size: int = 1,
    max_set_size: int = 10,
) -> list[_Population]:
    populations: list[_Population] = []
    for seed in seeds:
        rng = random.Random(seed)
        agreement_level = rng.uniform(0.0, 1.0)
        if mode == "single":
            rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_single_pairs(
                pool, n_samples, agreement_level, rng,
                sibling_index=sibling_index, concept_by_code=concept_by_code,
                negative_index=negative_index,
            )
        else:
            rater_a, rater_b, rater_a_concepts, rater_b_concepts = _generate_set_pairs(
                pool, n_samples, agreement_level, rng,
                sibling_index=sibling_index, concept_by_code=concept_by_code,
                negative_index=negative_index,
                min_set_size=min_set_size, max_set_size=max_set_size,
            )
        reference_scores = AgreementScores.get(
            rater_a_concepts, rater_b_concepts, distance_function=set_jaccard_distance
        )
        populations.append((
            agreement_level, seed, rater_a, rater_b, rater_a_concepts, rater_b_concepts, reference_scores,
        ))
    return populations


def _needs_prebatch(dist_fn: DistanceFunction) -> bool:
    if isinstance(dist_fn, BipartiteMatchingDistance):
        return True
    return isinstance(getattr(dist_fn, "_base", None), BipartiteMatchingDistance)


_TIMING_FIELDS = [
    "dataset", "mode", "negative_mode", "measure", "n_pairs", "n_prompts", "elapsed_seconds", "pairs_per_second",
]


def _prebatch_populations(
    dist_fn: DistanceFunction,
    populations: list[_Population],
    name: str,
    chance_distance_function: DistanceFunction | None = None,
    max_exact_chance_pairs: int | None = None,
    mc_target_samples: int | None = None,
    mc_shuffles: int | None = None,
    mc_min_shuffles: int | None = None,
    batch_size: int = 32,
) -> dict | None:
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    pairs: list[tuple[list[str], list[str]]] = []

    def _queue(a, b) -> None:
        key = (tuple(a), tuple(b))
        if key not in seen:
            seen.add(key)
            pairs.append((list(a), list(b)))

    for _, _, rater_a, rater_b, _, _, _ in populations:
        for a, b in zip(rater_a, rater_b):
            _queue(a, b)
            _queue(b, a)
        if chance_distance_function is None:
            rater_a_labels = [frozenset(a) for a in rater_a]
            rater_b_labels = [frozenset(b) for b in rater_b]
            pooled_labels = rater_a_labels + rater_b_labels
            cap = AgreementScores.chance_kwargs(
                max_exact_chance_pairs, mc_target_samples, mc_shuffles, mc_min_shuffles
            )
            kappa_chance_pairs, _ = AgreementScores.chance_pairs_and_weights(
                rater_a_labels, rater_b_labels, **cap
            )
            pi_chance_pairs, _ = AgreementScores.chance_pairs_and_weights(
                pooled_labels, pooled_labels, **cap
            )
            for a, b in kappa_chance_pairs + pi_chance_pairs:
                _queue(a, b)

    if not pairs:
        return None
    tqdm.write(f"  Pre-computing {name}: {len(pairs):,} unique label pairs (batch_size={batch_size}) ...")
    start = timeit.default_timer()
    dist_fn.batch(pairs, batch_size=batch_size)
    elapsed = timeit.default_timer() - start
    pairs_per_second = len(pairs) / elapsed
    n_prompts = getattr(dist_fn, "last_n_prompts", None)
    tqdm.write(
        f"  Pre-computing {name}: done in {elapsed:.2f}s ({pairs_per_second:,.1f} pairs/s"
        f"{f', {n_prompts:,} prompts' if n_prompts is not None else ''})"
    )
    return {
        "measure": name,
        "n_pairs": len(pairs),
        "n_prompts": n_prompts,
        "elapsed_seconds": elapsed,
        "pairs_per_second": pairs_per_second,
    }


def _run_measure_over_populations(
    rows: list[dict],
    timing_rows: list[dict],
    predictions_writer: csv.DictWriter | None,
    name: str,
    dist_fn: DistanceFunction,
    populations: list[_Population],
    dataset: str,
    mode: str,
    negative_mode: str,
    cost_matrix_writer: CostMatrixWriter | None = None,
    chance_distance_function: DistanceFunction | None = None,
    chance_measure_name: str = "self",
    max_exact_chance_pairs: int | None = None,
    mc_target_samples: int | None = None,
    mc_shuffles: int | None = None,
    mc_min_shuffles: int | None = None,
    reasoning: bool = False,
    batch_size: int = 32,
) -> None:
    if _needs_prebatch(dist_fn):
        timing = _prebatch_populations(
            dist_fn, populations, name,
            chance_distance_function=chance_distance_function,
            max_exact_chance_pairs=max_exact_chance_pairs,
            mc_target_samples=mc_target_samples,
            mc_shuffles=mc_shuffles,
            mc_min_shuffles=mc_min_shuffles,
            batch_size=batch_size,
        )
        if timing is not None:
            timing_rows.append({
                "dataset": dataset, "mode": mode, "negative_mode": negative_mode, **timing,
            })

    for (
        agreement_level, seed, rater_a, rater_b, rater_a_concepts, rater_b_concepts, truth,
    ) in tqdm(populations, desc=f"  {name}", leave=False, unit="pop"):
        try:
            scores_by_matching = AgreementScores.get_by_matching(
                rater_a, rater_b, distance_function=dist_fn,
                chance_distance_function=chance_distance_function or dist_fn,
                max_exact_chance_pairs=max_exact_chance_pairs,
                mc_target_samples=mc_target_samples,
                mc_shuffles=mc_shuffles,
                mc_min_shuffles=mc_min_shuffles,
            )
            sample_results = dist_fn.batch(
                [(frozenset(a), frozenset(b)) for a, b in zip(rater_a, rater_b)], batch_size=batch_size
            )
            for matching, scores in scores_by_matching.items():
                mae_avg_agreement, mape_avg_agreement = _mae_mape(
                    scores.average_agreement, truth.average_agreement
                )
                mae_kappa, mape_kappa = _mae_mape(scores.cohens_kappa, truth.cohens_kappa)
                mae_pi, mape_pi = _mae_mape(scores.scotts_pi, truth.scotts_pi)
                mae_sigma, mape_sigma = _mae_mape(scores.sigma, truth.sigma)
                mae_ks, mape_ks = _mae_mape(scores.ks, truth.ks)
                rows.append({
                    "dataset": dataset,
                    "mode": mode,
                    "negative_mode": negative_mode,
                    "agreement_level": agreement_level,
                    "seed": seed,
                    "measure": name,
                    "matching": matching,
                    "chance_measure": chance_measure_name,
                    "reasoning": reasoning,
                    "true_avg_agreement": truth.average_agreement,
                    "true_cohens_kappa": truth.cohens_kappa,
                    "true_scotts_pi": truth.scotts_pi,
                    "true_sigma": truth.sigma,
                    "true_ks": truth.ks,
                    "avg_agreement": scores.average_agreement,
                    "cohens_kappa": scores.cohens_kappa,
                    "scotts_pi": scores.scotts_pi,
                    "sigma": scores.sigma,
                    "ks": scores.ks,
                    "mae_avg_agreement": mae_avg_agreement,
                    "mape_avg_agreement": mape_avg_agreement,
                    "mae_kappa": mae_kappa,
                    "mape_kappa": mape_kappa,
                    "mae_pi": mae_pi,
                    "mape_pi": mape_pi,
                    "mae_sigma": mae_sigma,
                    "mape_sigma": mape_sigma,
                    "mae_ks": mae_ks,
                    "mape_ks": mape_ks,
                })
            if predictions_writer is not None:
                predicted_assignments = getattr(dist_fn, "last_assignments", None)
                greedy_assignments = getattr(dist_fn, "last_greedy_assignments", None)
                confident_assignments = getattr(dist_fn, "last_confident_assignments", None)
                cost_matrices = getattr(dist_fn, "last_cost_matrices", None)
                for i, (dist, a_i, b_i, a_concepts_i, b_concepts_i) in enumerate(
                    zip(sample_results, rater_a, rater_b, rater_a_concepts, rater_b_concepts)
                ):
                    a_concept_by_label = dict(zip(reversed(a_i), reversed(a_concepts_i)))
                    b_concept_by_label = dict(zip(reversed(b_i), reversed(b_concepts_i)))
                    a_i = _as_matching_str_list(frozenset(a_i))
                    b_i = _as_matching_str_list(frozenset(b_i))
                    a_concepts_i = [a_concept_by_label[s] for s in a_i]
                    b_concepts_i = [b_concept_by_label[s] for s in b_i]
                    true_assignment = _true_assignment(a_concepts_i, b_concepts_i)
                    predicted_assignment = (
                        predicted_assignments[i] if predicted_assignments is not None else None
                    )
                    confident_assignment = (
                        confident_assignments[i] if confident_assignments is not None else None
                    )
                    distance_oracle = (
                        dist_fn.score_with_assignment(i, true_assignment)
                        if hasattr(dist_fn, "score_with_assignment")
                        else None
                    )
                    distance_greedy = (
                        dist_fn.score_with_assignment(i, greedy_assignments[i])
                        if hasattr(dist_fn, "score_with_assignment") and greedy_assignments is not None
                        else None
                    )
                    if cost_matrix_writer is not None and cost_matrices is not None and cost_matrices[i]:
                        cost_matrix_writer.write(
                            {
                                "dataset": dataset,
                                "mode": mode,
                                "negative_mode": negative_mode,
                                "agreement_level": agreement_level,
                                "seed": seed,
                                "measure": name,
                                "sample_index": i,
                            },
                            cost_matrices[i],
                        )
                    predictions_writer.writerow({
                        "dataset": dataset,
                        "mode": mode,
                        "negative_mode": negative_mode,
                        "agreement_level": agreement_level,
                        "seed": seed,
                        "measure": name,
                        "chance_measure": chance_measure_name,
                        "reasoning": reasoning,
                        "sample_index": i,
                        "rater_a_set_size": len(a_i),
                        "rater_b_set_size": len(b_i),
                        "rater_a": json.dumps(a_i, ensure_ascii=False),
                        "rater_b": json.dumps(b_i, ensure_ascii=False),
                        "distance": dist,
                        "distance_oracle": distance_oracle,
                        "distance_greedy": distance_greedy,
                        "true_match": sorted(a_concepts_i) == sorted(b_concepts_i),
                        "true_assignment": json.dumps(true_assignment),
                        "predicted_assignment": (
                            json.dumps(predicted_assignment) if predicted_assignment is not None else None
                        ),
                        "greedy_assignment": (
                            json.dumps(greedy_assignments[i]) if greedy_assignments is not None else None
                        ),
                        "confident_assignment": (
                            json.dumps(confident_assignment) if confident_assignment is not None else None
                        ),
                    })
        except Exception as exc:
            tqdm.write(f"  [WARN] {name} failed (agreement={agreement_level:.1f}, seed={seed}): {exc}")


def run_experiment(
    concepts: dict[str, ConceptEntry],
    mode: str,
    negative_mode: str,
    measures: dict[str, DistanceFunction],
    n_samples: int,
    n_populations: int,
    base_seed: int,
    dataset: str,
    predictions_writer: csv.DictWriter | None = None,
    cost_matrix_writer: CostMatrixWriter | None = None,
    chance_distance_function: DistanceFunction | None = None,
    chance_measure_name: str = "self",
    max_exact_chance_pairs: int | None = None,
    mc_target_samples: int | None = None,
    mc_shuffles: int | None = None,
    mc_min_shuffles: int | None = None,
    reasoning: bool = False,
    min_set_size: int = 1,
    max_set_size: int = 10,
    parent_terms: dict[str, str] | None = None,
    ancestor_parent_codes: dict[str, list[str]] | None = None,
    negative_hierarchy_hops: int = 1,
    lexical_top_k: int = 10,
    batch_size: int = 32,
) -> tuple[list[dict], list[dict]]:
    concept_by_code = concepts
    rows: list[dict] = []
    timing_rows: list[dict] = []
    seeds = [base_seed + i for i in range(n_populations)]

    if mode == "single":
        measures = {
            name: fn for name, fn in measures.items() if not fn.is_bipartite
        }
    elif mode == "set":
        measures = {name: fn for name, fn in measures.items() if not name.endswith("_single")}

    sibling_index: dict[str, list[str]] | None = None
    negative_index: dict[str, list[ConceptEntry]] | None = None
    if negative_mode == "sibling":
        sibling_index = build_sibling_index(concepts)
    elif negative_mode in ("parent", "children"):
        negative_index = build_hierarchy_index(
            concepts, direction=negative_mode, parent_terms=parent_terms,
            ancestor_parent_codes=ancestor_parent_codes, hops=negative_hierarchy_hops,
        )
    elif negative_mode == "lexical":
        negative_index = build_lexical_index(concepts, top_k=lexical_top_k)

    def _pool() -> list[ConceptEntry]:
        if sibling_index is not None:
            return [c for c in concepts.values() if sibling_index[c.code]]
        if negative_index is not None:
            return [c for c in concepts.values() if negative_index[c.code]]
        return list(concepts.values())

    pool = _pool()
    if negative_mode == "sibling":
        print(f"  Sibling pool: {len(pool):,}/{len(concepts):,} concepts eligible.")
    elif negative_mode in ("parent", "children", "lexical"):
        print(f"  {negative_mode.capitalize()} pool: {len(pool):,}/{len(concepts):,} concepts eligible.")

    print(f"  Generating populations ({len(seeds)} population(s) with independently sampled agreement levels) ...")
    populations = _generate_populations(
        mode, pool, n_samples, seeds, sibling_index, concept_by_code,
        negative_index=negative_index,
        min_set_size=min_set_size, max_set_size=max_set_size,
    )

    for name, dist_fn in tqdm(measures.items(), desc=f"{mode} measures", unit="measure"):
        _run_measure_over_populations(
            rows, timing_rows, predictions_writer, name, dist_fn, populations, dataset, mode, negative_mode,
            cost_matrix_writer=cost_matrix_writer,
            chance_distance_function=chance_distance_function,
            chance_measure_name=chance_measure_name,
            max_exact_chance_pairs=max_exact_chance_pairs,
            mc_target_samples=mc_target_samples,
            mc_shuffles=mc_shuffles,
            mc_min_shuffles=mc_min_shuffles,
            reasoning=reasoning,
            batch_size=batch_size,
        )
        dist_fn.release()

    return rows, timing_rows


def _summary_key(row: dict) -> str:
    matching = row.get("matching", "none")
    return row["measure"] if matching in ("none", "hungarian") else f"{row['measure']} ({matching})"


def _print_error_table(rows: list[dict], mode: str, metric_label: str, mae_field: str, mape_field: str) -> None:
    mae_by_measure: dict[str, list[float]] = defaultdict(list)
    mape_by_measure: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        mae_by_measure[_summary_key(r)].append(r[mae_field])
        if not np.isnan(r[mape_field]):
            mape_by_measure[_summary_key(r)].append(r[mape_field])
    ranked = sorted(mae_by_measure.items(), key=lambda kv: np.mean(kv[1]))
    print(f"\n  Mean {metric_label} MAE by measure ({mode}):")
    for name, errs in ranked:
        mape_vals = mape_by_measure.get(name, [])
        mape_str = f"{np.mean(mape_vals):.2f}%" if mape_vals else "n/a"
        print(f"    {name:<30s}  {np.mean(errs):.4f} ± {np.std(errs):.4f}   MAPE={mape_str}")


def _significance_vs_best(
    rows: list[dict], mae_field: str, num_rounds: int = 1000, seed: int = 0,
) -> tuple[str, float, list[tuple[str, float, float]]]:
    errors_by_measure: dict[str, dict[int, float]] = defaultdict(dict)
    for r in rows:
        errors_by_measure[_summary_key(r)][r["seed"]] = r[mae_field]

    means = {name: float(np.mean(list(errs.values()))) for name, errs in errors_by_measure.items()}
    best = min(means, key=means.get)
    best_errors = errors_by_measure[best]

    results: list[tuple[str, float, float]] = []
    for name, errs in errors_by_measure.items():
        if name == best:
            continue
        common_seeds = sorted(set(errs) & set(best_errors))
        if len(common_seeds) < 2:
            results.append((name, means[name], float("nan")))
            continue
        x = [errs[s] for s in common_seeds]
        y = [best_errors[s] for s in common_seeds]
        p_value = permutation_test(
            x, y, method="approximate", num_rounds=num_rounds, seed=seed, paired=True,
        )
        results.append((name, means[name], p_value))

    results.sort(key=lambda t: t[1])
    return best, means[best], results


_SIGNIFICANCE_FIELDS = [
    "dataset", "mode", "negative_mode", "metric", "measure", "mean_mae", "p_value", "significant", "is_best", "best_measure",
]


def _significance_rows_for_metric(
    rows: list[dict], dataset: str, mode: str, negative_mode: str, metric_label: str, mae_field: str,
    alpha: float = 0.05,
) -> list[dict]:
    best, best_mean, results = _significance_vs_best(rows, mae_field)
    print(
        f"\n  Permutation test ({metric_label} MAE, paired, 1000 rounds) vs best measure "
        f"'{best}' (mean={best_mean:.4f}), {mode}:"
    )
    sig_rows = [{
        "dataset": dataset, "mode": mode, "negative_mode": negative_mode, "metric": metric_label,
        "measure": best, "mean_mae": best_mean, "p_value": float("nan"),
        "significant": False, "is_best": True, "best_measure": best,
    }]
    for name, mean_err, p_value in results:
        significant = (not np.isnan(p_value)) and p_value < alpha
        flag = "(< 2 shared populations)" if np.isnan(p_value) else ("significant" if significant else "not significant")
        print(f"    {name:<30s}  mean={mean_err:.4f}   p={p_value:.4f}   {flag}")
        sig_rows.append({
            "dataset": dataset, "mode": mode, "negative_mode": negative_mode, "metric": metric_label,
            "measure": name, "mean_mae": mean_err, "p_value": p_value,
            "significant": significant, "is_best": False, "best_measure": best,
        })
    return sig_rows


def _print_summary(rows: list[dict], dataset: str, mode: str, negative_mode: str) -> list[dict]:
    _print_error_table(rows, mode, "avg_agreement", "mae_avg_agreement", "mape_avg_agreement")
    sig_rows = _significance_rows_for_metric(rows, dataset, mode, negative_mode, "avg_agreement", "mae_avg_agreement")
    _print_error_table(rows, mode, "cohens_kappa", "mae_kappa", "mape_kappa")
    sig_rows += _significance_rows_for_metric(rows, dataset, mode, negative_mode, "cohens_kappa", "mae_kappa")
    _print_error_table(rows, mode, "scotts_pi", "mae_pi", "mape_pi")
    sig_rows += _significance_rows_for_metric(rows, dataset, mode, negative_mode, "scotts_pi", "mae_pi")
    _print_error_table(rows, mode, "sigma", "mae_sigma", "mape_sigma")
    sig_rows += _significance_rows_for_metric(rows, dataset, mode, negative_mode, "sigma", "mae_sigma")
    _print_error_table(rows, mode, "ks", "mae_ks", "mape_ks")
    sig_rows += _significance_rows_for_metric(rows, dataset, mode, negative_mode, "ks", "mae_ks")
    return sig_rows


def _build_measures(
    include_llm: bool, simple_only: bool, enable_reasoning: bool = False, llm_fp16: bool = False
) -> dict[str, DistanceFunction]:
    measures: dict[str, DistanceFunction] = {
        "exact": exact_match_distance,
        "jaccard": jaccard_distance,
        "jaccard_bipartite": bipartite_jaccard_distance,
        "masi": masi_distance,
        "levenshtein": levenshtein_distance,
        "levenshtein_bipartite": bipartite_levenshtein_distance,
        "normalized_indel_similarity": normalized_indel_similarity,
        "normalized_indel_similarity_bipartite": bipartite_normalized_indel_distance,
        "lcsseq": lcsseq_distance,
        "lcsseq_bipartite": bipartite_lcsseq_distance,
    }
    if simple_only:
        return measures

    measures.update({
        "embedding_pubmedbert": bipartite_embedding_distance,
        "embedding_qwen3": bipartite_qwen3_embedding_distance,
        "nli_mednli": bipartite_nli_distance,
        "nli_mdeberta": bipartite_nli_multilingual_distance,
        "embedding_pubmedbert_single": single_embedding_pubmedbert,
        "embedding_qwen3_single": single_embedding_qwen3,
        "nli_mednli_single": single_nli_mednli,
        "nli_mdeberta_single": single_nli_multilingual,
    })

    if include_llm:
        from soft_irr.evaluation.distances import (
            bipartite_medgemma_log_prob_distance,
            bipartite_medgemma_verbalized_prob_distance,
            bipartite_qwen35_log_prob_distance,
            bipartite_qwen35_verbalized_prob_distance,
            single_logprob_medgemma,
            single_logprob_qwen35,
            single_verbalized_medgemma,
            single_verbalized_qwen35,
        )
        for dist in (
            bipartite_medgemma_log_prob_distance,
            bipartite_medgemma_verbalized_prob_distance,
            bipartite_qwen35_log_prob_distance,
            bipartite_qwen35_verbalized_prob_distance,
        ):
            dist.set_reasoning(enable_reasoning)
            dist.set_precision(llm_fp16)
        measures["logprob_qwen35"] = bipartite_qwen35_log_prob_distance
        measures["logprob_medgemma"] = bipartite_medgemma_log_prob_distance
        measures["verbalized_qwen35"] = bipartite_qwen35_verbalized_prob_distance
        measures["verbalized_medgemma"] = bipartite_medgemma_verbalized_prob_distance
        measures["logprob_qwen35_single"] = single_logprob_qwen35
        measures["logprob_medgemma_single"] = single_logprob_medgemma
        measures["verbalized_qwen35_single"] = single_verbalized_qwen35
        measures["verbalized_medgemma_single"] = single_verbalized_medgemma
    return measures


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic IRR experiments with controlled agreement levels."
    )
    p.add_argument("--dataset", required=True, choices=["icd11", "meddra", "mesh"])
    p.add_argument(
        "--icd11-synonyms",
        default=str(_ROOT_DIR / "data/icd11/synonyms.json"),
        help="(ICD-11 only) JSON file mapping code \u2192 [synonym, ...].",
    )
    p.add_argument("--mode", default="both", choices=["single", "set", "both"])
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--min-set-size", type=int, default=2, help="(set mode) Minimum set size.")
    p.add_argument("--max-set-size", type=int, default=5, help="(set mode) Maximum set size.")
    p.add_argument("--n-populations", type=int, default=30,
                   help="Independent populations, each with its own sampled agreement level.")
    p.add_argument(
        "--negative-mode",
        default="random",
        choices=["random", "sibling", "parent", "children", "lexical"],
        help=(
            "How to sample disagreement pairs: 'random' any other concept, 'sibling' same parent, "
            "'parent' an ancestor (over-generalization), 'children' a child (over-specification, "
            "no-op for MedDRA), 'lexical' a string-similar concept. Concepts with no candidates "
            "are excluded from the sampling pool entirely; the eligible count is printed per run."
        ),
    )
    p.add_argument("--lexical-top-k", type=int, default=10,
                   help="(negative-mode=lexical) Nearest neighbors by Indel similarity to sample from.")
    p.add_argument(
        "--negative-hierarchy-hops", type=int, default=2,
        help="(negative-mode=parent) Hierarchy levels to pool candidates from.",
    )
    p.add_argument("--base-seed", type=int, default=52,
                   help="Base RNG seed; population i uses seed base_seed+i.")
    p.add_argument("--output", default="results/", help="Directory for output CSV files.")
    p.add_argument(
        "--include-llm",
        action="store_true",
        help="Also run slow LLM-based measures (logprob, verbalized). Requires GPU or API key.",
    )
    p.add_argument(
        "--enable-reasoning",
        action="store_true",
        help=(
            "(--include-llm only) Let local LLM measures reason freely before answering instead "
            "of force-completing into the answer slot. Slower."
        ),
    )
    p.add_argument(
        "--llm-fp16",
        action="store_true",
        help=(
            "(--include-llm only) Load medgemma in fp16 + eager attention instead of fp32 + SDPA. "
            "Faster on Turing, but fp16 has produced NaN logits here -- check with "
            "scripts/check_llm_precision.py first."
        ),
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for the model-backed measures' pre-computation pass.",
    )
    p.add_argument("--simple-only", action="store_true", help="Only calculate simple string distances")
    p.add_argument(
        "--chance-measure",
        default=None,
        help=(
            "Compute the chance term (Ae) with this measure instead of the measure under test. "
            "Large speedup, but Ae and Ao are then scored differently. E.g. levenshtein_bipartite."
        ),
    )
    p.add_argument(
        "--max-exact-chance-pairs",
        type=int,
        default=None,
        help=(
            "Marginal-pair budget above which Ae switches to its Monte Carlo estimate (default: "
            f"{DEFAULT_MAX_EXACT_CHANCE_PAIRS}). Lowering it stays unbiased, only adds variance."
        ),
    )
    p.add_argument(
        "--mc-target-samples",
        type=int,
        default=None,
        help=(
            f"Target MC sample count Ae scales its shuffles to hit (default: {DEFAULT_MC_TARGET_SAMPLES})."
        ),
    )
    p.add_argument(
        "--mc-shuffles",
        type=int,
        default=None,
        help="Pin Ae's Monte Carlo permutation count, overriding --mc-target-samples.",
    )
    p.add_argument(
        "--mc-min-shuffles",
        type=int,
        default=None,
        help=(
            f"Lower bound on Ae's derived permutation count (default: {DEFAULT_MC_MIN_SHUFFLES}). "
            "Only bites when --mc-target-samples <= this * n_samples."
        ),
    )
    p.add_argument(
        "--draft",
        action="store_true",
        help=(
            f"Shorthand for --max-exact-chance-pairs {_DRAFT_MAX_EXACT_CHANCE_PAIRS}. Ae is still "
            "scored with the measure under test, just Monte Carlo estimated. NOT publication-grade."
        ),
    )
    p.add_argument(
        "--no-cache",
        dest="use_pair_cache",
        action="store_false",
        help=(
            "Re-run all LLM/NLI inference instead of reusing the edge costs recorded in earlier "
            "runs' costs_*.jsonl.gz sidecars."
        ),
    )
    p.add_argument(
        "--cache-from",
        default=None,
        help="Directory of earlier runs to recover edge costs from (default: --output).",
    )
    p.add_argument(
        "--no-cost-matrices",
        dest="save_cost_matrices",
        action="store_false",
        help="Skip the costs_synthetic_*.jsonl.gz sidecar that makes offline re-scoring possible.",
    )
    return p.parse_args()


_DATA_PATH_DEFAULTS = {
    "icd11": str(_ROOT_DIR / "data/icd11/LinearizationMiniOutput-MMS-en.txt"),
    "meddra": str(_ROOT_DIR / "data/meddra/"),
    "mesh": str(_ROOT_DIR / "data/mesh/desc2026.xml"),
}


def main() -> None:
    args = _parse_args()
    data_path = _DATA_PATH_DEFAULTS[args.dataset]

    print(f"Loading {args.dataset} from {data_path} ...")
    if args.dataset == "icd11":
        concepts = load_icd11(data_path, synonyms_json=args.icd11_synonyms)
        if not concepts:
            print(
                "  [ERROR] No ICD-11 concepts loaded. The linearization TSV has no synonym "
                "columns. Provide --icd11-synonyms pointing to a JSON file mapping "
                "code → [synonym, ...] fetched from the WHO ICD-11 API."
            )
            return
    elif args.dataset == "meddra":
        concepts = load_meddra(data_path)
    else:
        concepts = load_mesh(data_path)

    print(f"  Loaded {len(concepts):,} concepts with at least one synonym.")
    if not concepts:
        print(
            f"  [ERROR] No {args.dataset} concepts loaded from {data_path}. The taxonomy source "
            "files are gitignored, so check they are actually present on this machine."
        )
        return

    parent_terms: dict[str, str] | None = None
    ancestor_parent_codes: dict[str, list[str]] | None = None
    if args.negative_mode == "parent":
        if args.dataset == "meddra":
            parent_terms = load_meddra_ancestor_names(data_path)
            ancestor_parent_codes = load_meddra_ancestor_hierarchy(data_path)
        elif args.dataset == "icd11":
            parent_terms = load_icd11_ancestor_titles(data_path)
            ancestor_parent_codes = load_icd11_ancestor_hierarchy(data_path)

    measures = _build_measures(
        args.include_llm, args.simple_only, args.enable_reasoning, args.llm_fp16
    )
    if args.use_pair_cache and wanted_measures(measures):
        cache_dir = args.cache_from or args.output
        stats: dict[str, int] = {}
        costs = load_costs(
            cache_dir, wanted_measures(measures), reasoning=args.enable_reasoning, stats=stats
        )
        attached = attach(measures, costs)
        print(
            f"\nEdge-cost cache: {sum(len(v) for v in costs.values()):,} pairs recovered from "
            f"{stats['files']:,} runs in {cache_dir}, backing {attached} measures"
        )
    modes = ["single", "set"] if args.mode == "both" else [args.mode]

    chance_measure_name, max_exact_chance_pairs = chance_settings(args)
    if chance_measure_name and chance_measure_name not in measures:
        raise SystemExit(
            f"--chance-measure {chance_measure_name!r} is not one of this run's measures: "
            + ", ".join(sorted(measures))
        )
    chance_distance_function = measures[chance_measure_name] if chance_measure_name else None
    if chance_measure_name:
        print(
            f"\n[ablation] Chance term computed with {chance_measure_name}, not the measure under test"
            + (f", Monte Carlo above {max_exact_chance_pairs} marginal pairs" if max_exact_chance_pairs else "")
            + ". Compare against a run without the flag; not a substitute for one."
        )
    elif max_exact_chance_pairs is not None:
        print(
            f"\n[ablation] Chance term scored with the measure under test, Monte Carlo estimated "
            f"above {max_exact_chance_pairs} marginal pairs. Unbiased, but noisier than the exact "
            "cross product."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        suffix = output_suffix(args, mode)
        print(f"\nRunning {mode} experiments ({args.n_populations} populations with independently sampled agreement levels, negative_mode={args.negative_mode}) ...")

        run_name = f"synthetic_{args.dataset}_{mode}_{args.negative_mode}{suffix}"
        predictions_path = out_dir / f"predictions_{run_name}.csv"
        cost_matrix_path = out_dir / f"costs_{run_name}.jsonl.gz"
        pair_cost_path = out_dir / f"paircosts_{run_name}.jsonl.gz"
        with open(predictions_path, "w", newline="", encoding="utf-8") as pred_f, CostMatrixWriter(
            cost_matrix_path if args.save_cost_matrices else None
        ) as cost_writer, PairCostWriter(
            pair_cost_path if args.save_cost_matrices else None
        ) as pair_cost_writer:
            attach_writer(measures, pair_cost_writer)
            predictions_writer = csv.DictWriter(pred_f, fieldnames=_PREDICTION_FIELDS)
            predictions_writer.writeheader()

            rows, timing_rows = run_experiment(
                concepts=concepts,
                mode=mode,
                negative_mode=args.negative_mode,
                measures=measures,
                n_samples=args.n_samples,
                n_populations=args.n_populations,
                base_seed=args.base_seed,
                dataset=args.dataset,
                predictions_writer=predictions_writer,
                cost_matrix_writer=cost_writer,
                chance_distance_function=chance_distance_function,
                chance_measure_name=chance_measure_name or "self",
                max_exact_chance_pairs=max_exact_chance_pairs,
                mc_target_samples=args.mc_target_samples,
                mc_shuffles=args.mc_shuffles,
                mc_min_shuffles=args.mc_min_shuffles,
                reasoning=args.enable_reasoning,
                min_set_size=args.min_set_size,
                max_set_size=args.max_set_size,
                parent_terms=parent_terms,
                ancestor_parent_codes=ancestor_parent_codes,
                negative_hierarchy_hops=args.negative_hierarchy_hops,
                lexical_top_k=args.lexical_top_k,
                batch_size=args.batch_size,
            )
        print(f"  Saved sample-level predictions → {predictions_path}")
        if cost_writer.n_written:
            print(f"  Saved {cost_writer.n_written:,} edge-cost matrices → {cost_matrix_path}")
        if pair_cost_writer.n_written:
            print(f"  Saved {pair_cost_writer.n_written:,} edge costs → {pair_cost_path}")

        timing_path = out_dir / f"timing_synthetic_{args.dataset}_{mode}_{args.negative_mode}{suffix}.csv"
        with open(timing_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_TIMING_FIELDS)
            writer.writeheader()
            writer.writerows(timing_rows)
        print(f"  Saved {len(timing_rows):,} precomputation timing rows → {timing_path}")

        out_path = out_dir / f"synthetic_{args.dataset}_{mode}_{args.negative_mode}{suffix}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {len(rows):,} rows → {out_path}")

        sig_rows = _print_summary(rows, args.dataset, mode, args.negative_mode)
        sig_path = out_dir / f"significance_synthetic_{args.dataset}_{mode}_{args.negative_mode}{suffix}.csv"
        with open(sig_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SIGNIFICANCE_FIELDS)
            writer.writeheader()
            writer.writerows(sig_rows)
        print(f"  Saved {len(sig_rows):,} significance rows → {sig_path}")


if __name__ == "__main__":
    main()
