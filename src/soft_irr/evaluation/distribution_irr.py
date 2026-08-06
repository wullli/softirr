
from __future__ import annotations

from collections.abc import Callable, Sequence
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import ArrayLike

DistanceFn = Callable[[Any, Any], float]
IndexPairs = tuple[np.ndarray, ...]
Labels = Sequence[Any] | np.ndarray


def halfwhere(whereM: IndexPairs) -> IndexPairs:
    half_indices = whereM[0] > whereM[1]
    return (whereM[0][half_indices], whereM[1][half_indices])


def dist_pardo(i: int, j: int, label_i: Any, label_j: Any, dist_fn: DistanceFn) -> tuple[int, int, float]:
    return i, j, (dist_fn(label_i, label_j) if i > j else np.nan)


def get_distance_matrix_singlethreaded(
    all_labels: Labels, dist_fn: DistanceFn, label_ij: IndexPairs | None = None
) -> np.ndarray:
    if label_ij is None:
        label_distances = [[dist_fn(label_a, label_b) if i < j else np.nan
                            for i, label_a in enumerate(all_labels)]
                            for j, label_b in enumerate(all_labels)]
        return np.array(label_distances)
    else:
        result = np.nan * np.ones((len(all_labels), len(all_labels)))
        for a, b in zip(label_ij[0], label_ij[1]):
            result[a, b] = dist_fn(all_labels[a], all_labels[b])
        return result


def get_distance_matrix(
    all_labels: Labels, dist_fn: DistanceFn, label_ij: IndexPairs | None = None
) -> np.ndarray:
    result = np.nan * np.ones((len(all_labels), len(all_labels)))

    if label_ij is None:
        label_range = range(len(all_labels))
        args = [(a, b, all_labels[a], all_labels[b], dist_fn) for b in label_range for a in label_range]
    else:
        args = [(a, b, all_labels[a], all_labels[b], dist_fn) for a, b in zip(label_ij[0], label_ij[1])]
    with Pool() as p:
        reduced = p.starmap(dist_pardo, args)

    for item in reduced:
        result[item[0], item[1]] = item[2]
    return result


def get_pair_sets(values: Labels, same_item_ij: IndexPairs) -> list[list[Any]]:
    result = np.nan * np.ones((len(values), len(values), 2))
    for a, b in zip(same_item_ij[0], same_item_ij[1]):
        result[a][b] = np.array([values[a], values[b]])
    vector = result[same_item_ij]
    return [list([pair[0], pair[1]]) for pair in vector]


class DoDa():

    def __init__(self, observed_distances: ArrayLike = (), expected_distances: ArrayLike = ()):
        self.observed_distances = np.asarray(observed_distances, dtype=float)
        self.expected_distances = np.asarray(expected_distances, dtype=float)

    def plot_distance_distributions(self, title: str | None = None, twinx: bool = False) -> None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.hist(self.observed_distances, color="b", alpha=0.5, bins=16)
        DoM, = ax.plot([self.observed_distances.mean(), self.observed_distances.mean()], [0, ax.get_ylim()[1]], "b:")
        ax2 = ax.twinx() if twinx else ax
        ax2.hist(self.expected_distances, color="r", alpha=0.5, bins=16)
        DeM, = ax2.plot([self.expected_distances.mean(), self.expected_distances.mean()], [0, ax2.get_ylim()[1]], "r:")
        plt.legend([DoM, DeM], ["observed distance", "expected distance"])
        if title is not None:
            plt.title(title)

    def get_krippendorff_alpha(self) -> float:
        return float(1 - self.observed_distances.mean() / self.expected_distances.mean())

    def get_sigma(self, thresh: float = 0.05, use_kde: bool = True, debug: bool = False) -> float:
        if use_kde:
            kde_De = stats.gaussian_kde(self.expected_distances)
            if debug:
                import matplotlib.pyplot as plt

                kde_Do = stats.gaussian_kde(self.observed_distances)
                plt.scatter(self.expected_distances, kde_De.pdf(self.expected_distances), color="r", alpha=0.5)
                plt.scatter(self.observed_distances, kde_Do.pdf(self.observed_distances), color="c", alpha=0.5)
                plt.show()
            pDeLtDo = np.array([kde_De.integrate_box_1d(0, d) for d in self.observed_distances])
            if debug:
                import matplotlib.pyplot as plt

                plt.scatter(self.observed_distances, pDeLtDo)
                self.plot_distance_distributions()
            frac_Do_above_thresh = np.mean(pDeLtDo >= thresh)
            return float(1 - frac_Do_above_thresh)
        else:
            bad_dist_thresh = np.quantile(self.expected_distances, thresh)
            frac_Do_above_thresh = np.mean(self.observed_distances >= bad_dist_thresh)
            return float(1 - frac_Do_above_thresh)

    def get_ks(self, fast: bool = True, debug: bool = False) -> float:
        d_o = self.observed_distances.flatten()
        d_e = self.expected_distances.flatten()
        if fast:
            return float(stats.ks_2samp(d_o, d_e, alternative="greater").statistic)
        else:
            return float(np.mean([1 - stats.ks_2samp([x], d_e, alternative="greater").pvalue for x in d_o]))


class InterAnnotatorAgreement(DoDa):
    @classmethod
    def create_from_experiment(cls, experiment: Any, distance_fn: DistanceFn | None = None) -> InterAnnotatorAgreement:
        fn = experiment.distance_fn if distance_fn is None else distance_fn
        golddict = getattr(experiment, "golddict", None)
        return cls(experiment.annodf, experiment.item_colname, experiment.uid_colname, experiment.label_colname,
                   fn, golddict)

    def __init__(self, annodf: pd.DataFrame, item_colname: str, uid_colname: str, label_colname: str,
                 distance_fn: DistanceFn, golddict: dict[Any, Any] | None = None):
        super().__init__()
        self.distance_fn = distance_fn
        self.annodf = annodf.sort_values(item_colname)
        self.annodf = self.annodf.rename(columns={item_colname:'item', uid_colname:"worker", label_colname:"label"})
        self.items = self.annodf["item"].values
        self.workers = self.annodf["worker"].values
        self.all_labels = self.annodf["label"].values
        self.golddict = golddict
        self.dist_from_gold: list[float] | None = None
        self.distance_matrix: np.ndarray = np.empty((0, 0))

    def __getstate__(self) -> dict[str, Any]:
        to_serialize = ['observed_distances', 'expected_distances', 'distance_matrix', 'annodf',
                        'items_of_distances', 'workers_of_distances']
        state = dict(self.__dict__)
        return {k:state.get(k) for k in to_serialize}

    def setup(self, subsample_expected_distances: bool = True, parallel_calc: bool = False,
              precomputed_observed_distances: np.ndarray | None = None) -> None:
        gdm = get_distance_matrix if parallel_calc else get_distance_matrix_singlethreaded

        same_item = np.array([[np.nan if item_a != item_b else 1 for item_a in self.items] for item_b in self.items])

        same_item_ij = halfwhere(np.where(~np.isnan(same_item)))
        print("Calculating same-item distances")
        same_item_distM = gdm(self.all_labels, self.distance_fn, label_ij=same_item_ij)
        self.observed_distances = same_item_distM[same_item_ij]
        self.items_of_distances = get_pair_sets(self.items, same_item_ij)
        self.workers_of_distances = get_pair_sets(self.workers, same_item_ij)

        if precomputed_observed_distances is not None:
            self.expected_distances = precomputed_observed_distances
        else:
            different_item = np.array([[np.nan if item_a == item_b else 1 for item_a in self.items]
                                       for item_b in self.items])
            different_item_ij = halfwhere(np.where(~np.isnan(different_item)))
            if subsample_expected_distances:
                nsample = min(len(same_item_ij[0]), len(different_item_ij[0]))
                sample_i = np.random.choice(np.arange(len(different_item_ij[0])), size=nsample)
                different_item_ij = (different_item_ij[0][sample_i], different_item_ij[1][sample_i])
                print("Calculating different-item distances")
                different_item_distM = gdm(self.all_labels, self.distance_fn, label_ij=different_item_ij)

                self.distance_matrix = np.nansum([same_item_distM, different_item_distM], axis=0)
            else:
                self.distance_matrix = gdm(self.all_labels, self.distance_fn)
            self.expected_distances = self.distance_matrix[different_item_ij]

    def plot_matrix(self, labels: Sequence[Any] | None = None, figsize: int = 8, title: str | None = None,
                    show_grid: bool = False) -> None:
        import matplotlib.pyplot as plt

        plt.subplots(figsize=(figsize, figsize))
        if labels is not None:
            for i, label in enumerate(self.all_labels):
                plt.annotate(label, (i, 0.15 + i), color="k")
        plt.imshow(self.distance_matrix, vmin=-0.1, vmax=1.5, cmap="plasma")
        ax = plt.gca()
        if show_grid:
            minor_grids = np.arange(-.5, self.distance_matrix.shape[0], 1)
            ax.set_xticks(minor_grids, minor=True)
            ax.set_yticks(minor_grids, minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if title is not None:
            plt.title(title)
        plt.show()


def split_items_by_gold_error(iaa: InterAnnotatorAgreement, items_per_split: int = 4) -> list[np.ndarray]:
    assert items_per_split >= 3
    if iaa.dist_from_gold is None:
        iaa.dist_from_gold = [iaa.distance_fn(label, iaa.golddict.get(item))
                              for item, label in zip(iaa.items, iaa.all_labels)]
    iaa.annodf["error"] = iaa.dist_from_gold

    error_sorted_items = iaa.annodf.groupby("item")["error"].mean().sort_values().index.values
    n_splits = int(len(error_sorted_items) / items_per_split)
    split_items = np.array_split(error_sorted_items, n_splits)
    return split_items


def split_iaa_by_item(iaa: InterAnnotatorAgreement, split_items: Sequence[np.ndarray]) -> list[InterAnnotatorAgreement]:
    mini_iaas = []
    for items in split_items:
        mini_df = iaa.annodf[iaa.annodf["item"].isin(items)]
        mini_iaa = InterAnnotatorAgreement(mini_df, "item", "worker", "label", iaa.distance_fn)
        mini_iaa.setup(parallel_calc=False, precomputed_observed_distances=iaa.expected_distances)
        mini_iaas.append(mini_iaa)
    return mini_iaas
