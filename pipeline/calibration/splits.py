"""
Split helpers for the first calibration study.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def _get_record_value(record: Any, name: str) -> Any:
    if hasattr(record, name):
        return getattr(record, name)
    if isinstance(record, dict):
        return record[name]
    raise TypeError("Record must expose '{}' as an attribute or dict key.".format(name))


def make_distance_range_label(true_distance_m: float) -> str:
    """Map a metric distance into the first-attempt coarse distance range."""
    distance = float(true_distance_m)
    if distance <= 60.0:
        return "near"
    if distance <= 100.0:
        return "mid"
    return "far"


def _make_stratum_key(record: Any) -> str:
    return "{}|{}|{}".format(
        make_distance_range_label(_get_record_value(record, "true_distance_m")),
        _get_record_value(record, "weather"),
        _get_record_value(record, "time_of_day"),
    )


def _allocate_test_counts(
    stratum_to_indices: Dict[str, List[int]],
    holdout_fraction: float,
) -> Dict[str, int]:
    total_count = sum(len(indices) for indices in stratum_to_indices.values())
    target_test_count = int(round(total_count * holdout_fraction))

    floor_counts: Dict[str, int] = {}
    remainder_rows: List[Tuple[float, str]] = []
    capped_total = 0
    for stratum_key, indices in stratum_to_indices.items():
        exact = len(indices) * holdout_fraction
        base_count = int(math.floor(exact))
        # Never consume every sample from a populated stratum if avoidable.
        base_count = min(base_count, max(0, len(indices) - 1))
        floor_counts[stratum_key] = base_count
        capped_total += base_count
        remainder_rows.append((exact - base_count, stratum_key))

    remaining = target_test_count - capped_total
    if remaining < 0:
        raise ValueError("Invalid split allocation produced too many test samples.")

    for _, stratum_key in sorted(remainder_rows, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        current = floor_counts[stratum_key]
        max_allowed = max(0, len(stratum_to_indices[stratum_key]) - 1)
        if current < max_allowed:
            floor_counts[stratum_key] = current + 1
            remaining -= 1

    if remaining != 0:
        raise ValueError(
            "Unable to allocate the requested holdout split with the available strata."
        )

    return floor_counts


def make_stratified_cv_folds(
    records: Sequence[Any],
    num_folds: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """Create deterministic CV folds using the first-attempt strata definition."""
    if not records:
        raise ValueError("Cannot split an empty record sequence.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    rng = random.Random(seed)
    stratum_to_indices: Dict[str, List[int]] = defaultdict(list)
    strata_by_index: Dict[int, str] = {}

    for index, record in enumerate(records):
        stratum_key = _make_stratum_key(record)
        stratum_to_indices[stratum_key].append(index)
        strata_by_index[index] = stratum_key

    fold_buckets: List[List[int]] = [[] for _ in range(num_folds)]
    all_indices = set(range(len(records)))
    for stratum_key, indices in sorted(stratum_to_indices.items()):
        shuffled_indices = list(indices)
        rng.shuffle(shuffled_indices)
        for offset, index in enumerate(shuffled_indices):
            fold_buckets[offset % num_folds].append(index)

    cv_folds: List[Dict[str, Any]] = []
    for fold_index, val_indices in enumerate(fold_buckets):
        val_sorted = sorted(val_indices)
        val_set = set(val_sorted)
        train_sorted = sorted(all_indices - val_set)
        cv_folds.append(
            {
                "fold_index": fold_index,
                "train_indices": train_sorted,
                "val_indices": val_sorted,
            }
        )

    return {
        "seed": seed,
        "num_folds": num_folds,
        "cv_folds": cv_folds,
        "strata_by_index": {str(index): stratum for index, stratum in strata_by_index.items()},
    }


def make_fixed_holdout_and_cv_splits(
    records: Sequence[Any],
    holdout_fraction: float = 0.15,
    num_folds: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Create a deterministic held-out test split plus CV folds on the dev split.
    """
    if not records:
        raise ValueError("Cannot split an empty record sequence.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2.")

    rng = random.Random(seed)
    stratum_to_indices: Dict[str, List[int]] = defaultdict(list)
    strata_by_index: Dict[int, str] = {}

    for index, record in enumerate(records):
        stratum_key = _make_stratum_key(record)
        stratum_to_indices[stratum_key].append(index)
        strata_by_index[index] = stratum_key

    for indices in stratum_to_indices.values():
        rng.shuffle(indices)

    test_counts = _allocate_test_counts(stratum_to_indices, holdout_fraction=holdout_fraction)
    test_indices: List[int] = []
    dev_indices: List[int] = []
    dev_stratum_to_indices: Dict[str, List[int]] = {}

    for stratum_key, indices in stratum_to_indices.items():
        cutoff = test_counts[stratum_key]
        stratum_test = indices[:cutoff]
        stratum_dev = indices[cutoff:]
        test_indices.extend(stratum_test)
        dev_indices.extend(stratum_dev)
        dev_stratum_to_indices[stratum_key] = list(stratum_dev)

    test_indices = sorted(test_indices)
    dev_indices = sorted(dev_indices)
    dev_index_set = set(dev_indices)

    fold_buckets: List[List[int]] = [[] for _ in range(num_folds)]
    for stratum_key, indices in sorted(dev_stratum_to_indices.items()):
        shuffled_indices = list(indices)
        rng.shuffle(shuffled_indices)
        for offset, index in enumerate(shuffled_indices):
            fold_buckets[offset % num_folds].append(index)

    cv_folds: List[Dict[str, Any]] = []
    for fold_index, val_indices in enumerate(fold_buckets):
        val_sorted = sorted(val_indices)
        val_set = set(val_sorted)
        train_sorted = sorted(dev_index_set - val_set)
        cv_folds.append(
            {
                "fold_index": fold_index,
                "train_indices": train_sorted,
                "val_indices": val_sorted,
            }
        )

    return {
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "num_folds": num_folds,
        "test_indices": test_indices,
        "dev_indices": dev_indices,
        "cv_folds": cv_folds,
        "strata_by_index": {str(index): stratum for index, stratum in strata_by_index.items()},
    }
