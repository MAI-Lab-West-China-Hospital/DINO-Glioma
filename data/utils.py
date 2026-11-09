from pathlib import Path
from monai.data import Dataset
import json
import math
import os
import random
import re

import pandas as pd

try:
    from .dataset_configs import DATASET_CONFIGS
except ImportError:
    from dataset_configs import DATASET_CONFIGS

_REQUIRED_DATASET_CONFIG_KEYS = {
    "data_subdir",
    "labels_csv",
    "id_col",
    "seg_label_type_col",
    "task_columns",
    "modality_suffixes",
    "seg_suffix",
    "enabled_tasks",
}
_REQUIRED_MODALITIES = ("flair", "t1", "t1ce", "t2")

def _normalize_tasks(tasks, default_tasks):
    if tasks is None:
        normalized = list(default_tasks)
    elif isinstance(tasks, str):
        normalized = [tasks]
    else:
        normalized = list(tasks)

    if not normalized:
        raise ValueError("tasks must not be empty")
    if any(not isinstance(task, str) or not task.strip() for task in normalized):
        raise ValueError("Every task must be a non-empty string")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"tasks contains duplicate entries: {normalized}")

    return normalized

def _get_dataset_config(dataset_name):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Available datasets: {sorted(DATASET_CONFIGS)}"
        )

    config = DATASET_CONFIGS[dataset_name]
    _validate_dataset_config(dataset_name, config)
    return config

def _validate_dataset_config(dataset_name, config):
    missing_keys = _REQUIRED_DATASET_CONFIG_KEYS - set(config)
    if missing_keys:
        raise ValueError(
            f"Dataset configuration '{dataset_name}' is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    task_columns = config["task_columns"]
    enabled_tasks = config["enabled_tasks"]
    modality_suffixes = config["modality_suffixes"]
    subtype_col = config.get("subtype_col")
    enabled_subtype = config.get("enabled_subtype", False)

    if not isinstance(task_columns, dict):
        raise ValueError(f"Dataset '{dataset_name}' task_columns must be a mapping")
    if not isinstance(enabled_tasks, dict):
        raise ValueError(f"Dataset '{dataset_name}' enabled_tasks must be a mapping")
    if not isinstance(modality_suffixes, dict):
        raise ValueError(f"Dataset '{dataset_name}' modality_suffixes must be a mapping")
    if subtype_col is not None and (not isinstance(subtype_col, str) or not subtype_col):
        raise ValueError(f"Dataset '{dataset_name}' subtype_col must be a non-empty string")
    if not isinstance(enabled_subtype, bool):
        raise ValueError(f"Dataset '{dataset_name}' enabled_subtype must be a boolean")
    if enabled_subtype and subtype_col is None:
        raise ValueError(f"Dataset '{dataset_name}' must define subtype_col when enabled_subtype=True")

    missing_modalities = set(_REQUIRED_MODALITIES) - set(modality_suffixes)
    if missing_modalities:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing modality suffixes: "
            f"{sorted(missing_modalities)}"
        )

    for task, column in task_columns.items():
        if not isinstance(task, str) or not task:
            raise ValueError(f"Dataset '{dataset_name}' has an invalid task name: {task!r}")
        if not isinstance(column, str) or not column:
            raise ValueError(
                f"Dataset '{dataset_name}' has an invalid CSV column for task '{task}'"
            )
        if task not in enabled_tasks or not isinstance(enabled_tasks[task], bool):
            raise ValueError(
                f"Dataset '{dataset_name}' must define a boolean enabled_tasks "
                f"value for task '{task}'"
            )

    for key in _REQUIRED_MODALITIES:
        if not isinstance(modality_suffixes[key], str) or not modality_suffixes[key]:
            raise ValueError(
                f"Dataset '{dataset_name}' has an invalid suffix for modality '{key}'"
            )

    for key in (
        "data_subdir",
        "labels_csv",
        "id_col",
        "seg_label_type_col",
        "seg_suffix",
    ):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(
                f"Dataset '{dataset_name}' has an invalid configuration value for '{key}'"
            )

def _resolve_dataset_path(total_root, configured_path):
    path = Path(configured_path)
    return path if path.is_absolute() else Path(total_root) / path

def _load_label_table(labels_csv, id_col, seg_label_type_col, task_columns, subtype_col=None):
    """Load labels as strings, then validate and normalize configured task columns."""
    labels_csv = Path(labels_csv)
    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels_csv does not exist or is not a file: {labels_csv}")

    labels_df = pd.read_csv(labels_csv, dtype=str)
    if id_col not in labels_df.columns:
        raise ValueError(f"id_col '{id_col}' does not exist in labels_csv")
    if seg_label_type_col not in labels_df.columns:
        raise ValueError(
            f"seg_label_type column '{seg_label_type_col}' does not exist in labels_csv"
        )

    missing_columns = sorted(set(task_columns.values()) - set(labels_df.columns))
    if missing_columns:
        raise ValueError(
            f"Configured task columns do not exist in labels_csv: {missing_columns}"
        )

    allowed_labels = {-1, 0, 1}
    for task, column in task_columns.items():
        numeric_labels = pd.to_numeric(labels_df[column], errors="coerce")
        invalid_mask = numeric_labels.isna() | ~numeric_labels.isin(allowed_labels)
        if invalid_mask.any():
            invalid_values = labels_df.loc[invalid_mask, column].unique().tolist()
            raise ValueError(
                f"Invalid labels in task '{task}' (CSV column '{column}'). "
                f"Allowed values are -1, 0, and 1. Found: {invalid_values}"
            )
        labels_df[column] = numeric_labels.astype(int)

    if subtype_col is not None and subtype_col in labels_df.columns:
        raw_subtypes = labels_df[subtype_col]
        present_mask = raw_subtypes.notna() & raw_subtypes.str.strip().ne("")
        numeric_subtypes = pd.to_numeric(raw_subtypes.where(present_mask), errors="coerce")
        invalid_mask = present_mask & (numeric_subtypes.isna() | (numeric_subtypes < -1) | (numeric_subtypes % 1 != 0))
        if invalid_mask.any():
            invalid_values = raw_subtypes.loc[invalid_mask].unique().tolist()
            raise ValueError(
                f"Invalid WHO subtype labels in CSV column '{subtype_col}'. "
                f"Expected non-negative integer class labels, -1, or missing values. Found: {invalid_values}"
            )
        labels_df[subtype_col] = numeric_subtypes.fillna(-1).astype(int)

    return labels_df

def _build_task_labels_and_masks(row, tasks, task_columns, enabled_tasks):
    labels = []
    label_masks = []
    for task in tasks:
        column = task_columns.get(task)
        if column is None:
            labels.append(-1)
            label_masks.append(0)
            continue

        value = int(row[column])
        labels.append(value)
        label_masks.append(1 if value in (0, 1) and enabled_tasks[task] else 0)

    return labels, label_masks

def _get_seg_label_type(row, patient_id, column):
    raw_value = row[column]
    if pd.isna(raw_value) or not str(raw_value).strip():
        raise ValueError(
            f"Missing segmentation label type for patient_id {patient_id} "
            f"in CSV column '{column}'"
        )

    seg_label_type = str(raw_value).strip().lower()
    if seg_label_type not in {"manual", "pseudo"}:
        raise ValueError(
            f"Invalid segmentation label type '{seg_label_type}' for patient_id "
            f"{patient_id} in CSV column '{column}'. Expected 'manual' or 'pseudo'."
        )
    return seg_label_type

def _resolve_sample_paths(patient_dir, modality_suffixes, seg_suffix, patient_id):
    seg_files = sorted(patient_dir.glob(f"*{seg_suffix}"))
    if not seg_files:
        print(f"Warning: patient_id {patient_id}: no segmentation file found, skipping")
        return None
    if len(seg_files) > 1:
        print(
            f"Warning: patient_id {patient_id}: multiple segmentation files found, "
            f"using {seg_files[0].name}"
        )

    seg = seg_files[0]
    special_seg_suffix = "_GlistrBoost_ManuallyCorrected.nii.gz"
    # prefix = seg.name[: -len(seg_suffix)]
    # if prefix.endswith("_GlistrBoost_ManuallyCorrected.nii.gz"):
    if seg.name.endswith(special_seg_suffix):
        prefix = seg.name[:-len(special_seg_suffix)]
        paths = {
            modality: patient_dir / f"{prefix}{modality_suffixes[modality]}"
            for modality in _REQUIRED_MODALITIES
        }
    else:
        paths = {
            modality: patient_dir / f"{patient_id}{modality_suffixes[modality]}"
            for modality in _REQUIRED_MODALITIES
        }
    paths["seg"] = seg

    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        print(f"Warning: patient_id {patient_id}: missing files {missing_files}, skipping")
        return None

    empty_files = [name for name, path in paths.items() if path.stat().st_size == 0]
    if empty_files:
        print(f"Warning: patient_id {patient_id}: zero-byte files {empty_files}, skipping")
        return None

    return paths

def generate_datalist(total_root, dataset_name, tasks=None):
    """Generate one configured dataset's files for development or external testing."""
    config = _get_dataset_config(dataset_name)
    tasks = _normalize_tasks(tasks, config["task_columns"])
    data_dir = _resolve_dataset_path(total_root, config["data_subdir"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir does not exist or is not a directory: {data_dir}")
    configured_task_columns = {
        task: config["task_columns"][task]
        for task in tasks
        if task in config["task_columns"]
    }
    labels_df = _load_label_table(
        _resolve_dataset_path(total_root, config["labels_csv"]),
        config["id_col"],
        config["seg_label_type_col"],
        configured_task_columns,
        config.get("subtype_col"),
    )

    data_list = []
    for row_idx, row in labels_df.iterrows():
        if pd.isna(row[config["id_col"]]):
            print(f"Warning: row {row_idx}: missing patient_id, skipping")
            continue

        patient_id = str(row[config["id_col"]]).strip()
        if not patient_id:
            print(f"Warning: row {row_idx}: empty patient_id, skipping")
            continue

        seg_label_type = _get_seg_label_type(
            row,
            patient_id,
            config["seg_label_type_col"],
        )
        label, label_mask = _build_task_labels_and_masks(
            row,
            tasks,
            config["task_columns"],
            config["enabled_tasks"],
        )
        subtype_col = config.get("subtype_col")
        subtype = int(row[subtype_col]) if subtype_col in labels_df.columns else -1
        subtype_mask = int(config.get("enabled_subtype", False) and subtype >= 0)

        patient_dir = data_dir / patient_id
        if not patient_dir.is_dir():
            print(
                f"Warning: patient_id {patient_id}: "
                "no matched patient directory found, skipping"
            )
            continue

        paths = _resolve_sample_paths(
            patient_dir,
            config["modality_suffixes"],
            config["seg_suffix"],
            patient_id,
        )
        if paths is None:
            continue

        sample = {
            **{key: str(path) for key, path in paths.items()},
            "label": label,
            "label_mask": label_mask,
            "subtype": subtype,
            "subtype_mask": subtype_mask,
            "patient_id": patient_id,
            "patient_dir": patient_dir.name,
            "dataset": dataset_name,
            "seg_label_type": seg_label_type,
        }
        data_list.append(sample)

    return data_list

# -----------------------------------------------------------------------------------------------------------------------------------------
# Index the dataset
# -----------------------------------------------------------------------------------------------------------------------------------------
class IndexedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["index"] = idx
        return sample

    def __len__(self):
        return len(self.base)

# -----------------------------------------------------------------------------------------------------------------------------------------
# Split data for multi-task && deal with the baselinedata and FUdata
# -----------------------------------------------------------------------------------------------------------------------------------------
def _sample_sort_identifier(sample):
    for key in (
        "patient_dir",
        "seg",
    ):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return str(sample)

def _is_followup_sample(sample):
    identifier = sample.get("patient_dir") or sample.get("sample_id")

    if identifier is None:
        identifier = _sample_sort_identifier(sample)

    identifier = str(identifier).upper()

    return (
        re.search(r"(^|[_-])FU\d*", identifier) is not None
        or "FOLLOW" in identifier
    )

def select_representative_sample(samples):
    # Prefer baseline / non-follow-up sample.
    sorted_samples = sorted(samples, key=_sample_sort_identifier)

    for sample in sorted_samples:
        if not _is_followup_sample(sample):
            return sample

    return sorted_samples[0]

def _is_task_label_valid(label, task_idx, label_mask=None):
    """Use label_mask when available, with a fallback for older saved samples."""
    if label_mask is not None:
        return label_mask[task_idx] == 1
    return label[task_idx] in (0, 1)


def _sample_task_is_valid(sample, task_idx):
    return _is_task_label_valid(
        sample["label"],
        task_idx,
        sample.get("label_mask"),
    )

def _patient_task_is_valid(
    patient_id,
    task_idx,
    patient_to_label,
    patient_to_label_mask=None,
):
    label_mask = None
    if patient_to_label_mask is not None:
        label_mask = patient_to_label_mask.get(patient_id)
    return _is_task_label_valid(
        patient_to_label[patient_id],
        task_idx,
        label_mask,
    )

def _patient_label_stats(
    patient_ids,
    patient_to_label,
    num_tasks,
    patient_to_label_mask=None,
):
    stats = []

    for task_idx in range(num_tasks):
        valid = 0
        positive = 0
        negative = 0
        missing = 0

        for patient_id in patient_ids:
            label = patient_to_label[patient_id][task_idx]

            if not _patient_task_is_valid(
                patient_id,
                task_idx,
                patient_to_label,
                patient_to_label_mask,
            ):
                missing += 1
            else:
                valid += 1
                if label == 1:
                    positive += 1
                elif label == 0:
                    negative += 1

        stats.append({
            "valid": valid,
            "positive": positive,
            "negative": negative,
            "missing": missing,
        })

    return stats

def _positive_rates(
    patient_ids,
    patient_to_label,
    num_tasks,
    patient_to_label_mask=None,
):
    rates = []

    for task_idx in range(num_tasks):
        valid = 0
        positive = 0

        for patient_id in patient_ids:
            label = patient_to_label[patient_id][task_idx]
            if not _patient_task_is_valid(
                patient_id,
                task_idx,
                patient_to_label,
                patient_to_label_mask,
            ):
                continue

            valid += 1
            if label == 1:
                positive += 1

        rates.append(None if valid == 0 else positive / valid)

    return rates

def _valid_rates(
    patient_ids,
    patient_to_label,
    num_tasks,
    patient_to_label_mask=None,
):
    if len(patient_ids) == 0:
        return [None] * num_tasks

    rates = []

    for task_idx in range(num_tasks):
        valid = 0

        for patient_id in patient_ids:
            if _patient_task_is_valid(
                patient_id,
                task_idx,
                patient_to_label,
                patient_to_label_mask,
            ):
                valid += 1

        rates.append(valid / len(patient_ids))

    return rates

def _subtype_distribution(patient_ids, patient_to_subtype, patient_to_subtype_mask):
    counts = {}
    for patient_id in patient_ids:
        if patient_to_subtype_mask.get(patient_id, 0) != 1:
            continue
        subtype = patient_to_subtype[patient_id]
        counts[subtype] = counts.get(subtype, 0) + 1
    counts = dict(sorted(counts.items()))
    total = sum(counts.values())
    return counts, {subtype: count / total for subtype, count in counts.items()} if total else {}

def _print_split_statistics(
    name,
    files,
    patient_ids,
    patient_to_label,
    num_tasks,
    patient_to_label_mask=None,
    patient_to_subtype=None,
    patient_to_subtype_mask=None,
):
    print(f"{name}: {len(files)} samples, {len(patient_ids)} patients")

    stats = _patient_label_stats(
        patient_ids,
        patient_to_label,
        num_tasks,
        patient_to_label_mask,
    )
    for task_idx, task_stats in enumerate(stats):
        print(
            f"  task {task_idx}: "
            f"valid={task_stats['valid']}, "
            f"positive={task_stats['positive']}, "
            f"negative={task_stats['negative']}, "
            f"missing={task_stats['missing']}"
        )

    if patient_to_subtype is not None and patient_to_subtype_mask is not None and any(
        mask == 1 for mask in patient_to_subtype_mask.values()
    ):
        subtype_counts, _ = _subtype_distribution(patient_ids, patient_to_subtype, patient_to_subtype_mask)
        valid = sum(subtype_counts.values())
        print(f"  WHO subtype: valid={valid}, missing={len(patient_ids) - valid}, class_counts={subtype_counts}")

def split_by_patient_multitask_balance(
    all_files,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
    num_trials=5000,
    eval_one_sample_per_patient=True,
):
    if not all_files:
        return [], [], []

    total_ratio = train_size + val_size + test_size
    if total_ratio <= 0:
        raise ValueError("train_size + val_size + test_size must be positive.")

    train_ratio = train_size / total_ratio
    val_ratio = val_size / total_ratio
    test_ratio = test_size / total_ratio

    patient_to_samples = {}
    for sample in all_files:
        patient_id = sample["patient_id"]
        patient_to_samples.setdefault(patient_id, []).append(sample)

    for patient_id, samples in patient_to_samples.items():
        first_label = samples[0]["label"]
        first_mask = [
            1 if _sample_task_is_valid(samples[0], task_idx) else 0
            for task_idx in range(len(first_label))
        ]
        first_subtype = samples[0].get("subtype", -1)
        first_subtype_mask = samples[0].get("subtype_mask", 0)

        for sample in samples[1:]:
            sample_mask = [
                1 if _sample_task_is_valid(sample, task_idx) else 0
                for task_idx in range(len(sample["label"]))
            ]
            if sample["label"] != first_label or sample_mask != first_mask:
                sample_identifier = (
                    sample.get("patient_dir")
                    or sample.get("sample_id")
                    or _sample_sort_identifier(sample)
                )
                raise ValueError(
                    f"Inconsistent labels or label masks for patient_id {patient_id}: "
                    f"label={first_label}, mask={first_mask} vs "
                    f"label={sample['label']}, mask={sample_mask} "
                    f"for sample {sample_identifier}"
                )
            sample_subtype = sample.get("subtype", -1)
            sample_subtype_mask = sample.get("subtype_mask", 0)
            if sample_subtype != first_subtype or sample_subtype_mask != first_subtype_mask:
                raise ValueError(
                    f"Inconsistent WHO subtype or subtype mask for patient_id {patient_id}: "
                    f"subtype={first_subtype}, mask={first_subtype_mask} vs "
                    f"subtype={sample_subtype}, mask={sample_subtype_mask}"
                )

    patient_ids = sorted(patient_to_samples.keys(), key=str)
    num_patients = len(patient_ids)

    if num_patients < 3:
        raise ValueError(
            f"Need at least 3 unique patients for train/val/test split, got {num_patients}"
        )

    patient_to_label = {
        patient_id: patient_to_samples[patient_id][0]["label"]
        for patient_id in patient_ids
    }
    patient_to_label_mask = {
        patient_id: patient_to_samples[patient_id][0].get("label_mask")
        for patient_id in patient_ids
    }
    patient_to_subtype = {patient_id: patient_to_samples[patient_id][0].get("subtype", -1) for patient_id in patient_ids}
    patient_to_subtype_mask = {patient_id: patient_to_samples[patient_id][0].get("subtype_mask", 0) for patient_id in patient_ids}

    num_tasks = len(patient_to_label[patient_ids[0]])

    target_train_n = int(round(num_patients * train_ratio))
    target_val_n = int(round(num_patients * val_ratio))

    target_train_n = max(1, min(target_train_n, num_patients))
    target_val_n = max(1, min(target_val_n, num_patients - target_train_n))

    if target_train_n + target_val_n >= num_patients:
        target_val_n = max(0, num_patients - target_train_n)

    global_positive_rates = _positive_rates(
        patient_ids,
        patient_to_label,
        num_tasks,
        patient_to_label_mask,
    )
    global_valid_rates = _valid_rates(
        patient_ids,
        patient_to_label,
        num_tasks,
        patient_to_label_mask,
    )
    global_subtype_counts, global_subtype_distribution = _subtype_distribution(
        patient_ids, patient_to_subtype, patient_to_subtype_mask
    )
    min_split_ratio = min(ratio for ratio in (train_ratio, val_ratio, test_ratio) if ratio > 0)
    min_reliable_class_count = max(3, math.ceil(1 / min_split_ratio))
    for subtype, count in global_subtype_counts.items():
        if count < min_reliable_class_count:
            print(
                f"Warning: WHO subtype class {subtype} has only {count} valid patients; "
                f"fewer than {min_reliable_class_count} is too rare for reliable representation in all three splits."
            )

    rng = random.Random(random_state)
    best_score = None
    best_split = None

    for _ in range(num_trials):
        shuffled_ids = patient_ids[:]
        rng.shuffle(shuffled_ids)

        train_ids = shuffled_ids[:target_train_n]
        val_ids = shuffled_ids[target_train_n:target_train_n + target_val_n]
        test_ids = shuffled_ids[target_train_n + target_val_n:]

        split_ids = [train_ids, val_ids, test_ids]
        target_ratios = [train_ratio, val_ratio, test_ratio]

        score = 0.0

        for split_name, ids, target_ratio in zip(("train", "val", "test"), split_ids, target_ratios):
            actual_ratio = len(ids) / num_patients
            score += 0.05 * abs(actual_ratio - target_ratio)

            split_positive_rates = _positive_rates(
                ids,
                patient_to_label,
                num_tasks,
                patient_to_label_mask,
            )
            split_valid_rates = _valid_rates(
                ids,
                patient_to_label,
                num_tasks,
                patient_to_label_mask,
            )

            for global_positive_rate, split_positive_rate in zip(
                global_positive_rates,
                split_positive_rates,
            ):
                if global_positive_rate is None:
                    continue

                if split_positive_rate is None:
                    score += 1.0
                else:
                    score += abs(split_positive_rate - global_positive_rate)

            for global_valid_rate, split_valid_rate in zip(
                global_valid_rates,
                split_valid_rates,
            ):
                if global_valid_rate is None:
                    continue

                if split_valid_rate is None:
                    score += 1.0
                else:
                    score += 0.3 * abs(split_valid_rate - global_valid_rate)

            if global_subtype_distribution:
                split_subtype_counts, split_subtype_distribution = _subtype_distribution(
                    ids, patient_to_subtype, patient_to_subtype_mask
                )
                if not split_subtype_distribution:
                    score += 1.0
                else:
                    score += sum(
                        abs(split_subtype_distribution.get(subtype, 0.0) - global_rate)
                        for subtype, global_rate in global_subtype_distribution.items()
                    ) / len(global_subtype_distribution)
                if split_name in {"val", "test"}:
                    score += sum(
                        1.0 for subtype, count in global_subtype_counts.items()
                        if count >= min_reliable_class_count and subtype not in split_subtype_counts
                    )

        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_ids, val_ids, test_ids)

    print(f"Best split score: {best_score:.4f}")

    train_patient_ids, val_patient_ids, test_patient_ids = best_split

    if not train_patient_ids or not val_patient_ids or not test_patient_ids:
        raise ValueError(
            f"Empty split detected: "
            f"train={len(train_patient_ids)}, "
            f"val={len(val_patient_ids)}, "
            f"test={len(test_patient_ids)}. "
            f"Please adjust train_size/val_size/test_size or provide more patients."
        )

    train_files = []
    for patient_id in train_patient_ids:
        train_files.extend(patient_to_samples[patient_id])

    if eval_one_sample_per_patient:
        val_files = [
            select_representative_sample(patient_to_samples[patient_id])
            for patient_id in val_patient_ids
        ]
        test_files = [
            select_representative_sample(patient_to_samples[patient_id])
            for patient_id in test_patient_ids
        ]
    else:
        val_files = []
        for patient_id in val_patient_ids:
            val_files.extend(patient_to_samples[patient_id])

        test_files = []
        for patient_id in test_patient_ids:
            test_files.extend(patient_to_samples[patient_id])

    train_ids = {d["patient_id"] for d in train_files}
    val_ids = {d["patient_id"] for d in val_files}
    test_ids = {d["patient_id"] for d in test_files}

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    assert not overlap_train_val, f"Leakage: train ∩ val = {sorted(overlap_train_val)}"
    assert not overlap_train_test, f"Leakage: train ∩ test = {sorted(overlap_train_test)}"
    assert not overlap_val_test, f"Leakage: val ∩ test = {sorted(overlap_val_test)}"

    _print_split_statistics(
        "train",
        train_files,
        train_ids,
        patient_to_label,
        num_tasks,
        patient_to_label_mask,
        patient_to_subtype,
        patient_to_subtype_mask,
    )
    _print_split_statistics(
        "val",
        val_files,
        val_ids,
        patient_to_label,
        num_tasks,
        patient_to_label_mask,
        patient_to_subtype,
        patient_to_subtype_mask,
    )
    _print_split_statistics(
        "test",
        test_files,
        test_ids,
        patient_to_label,
        num_tasks,
        patient_to_label_mask,
        patient_to_subtype,
        patient_to_subtype_mask,
    )

    return train_files, val_files, test_files

def build_multidataset_splits(
    total_root,
    dataset_names,
    tasks=None,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    random_state=42,
    num_trials=5000,
    eval_one_sample_per_patient=True,
):
    """Split each development dataset independently, then concatenate by split."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    else:
        dataset_names = list(dataset_names)

    if not dataset_names:
        raise ValueError("dataset_names must not be empty")

    first_config = _get_dataset_config(dataset_names[0])
    normalized_tasks = _normalize_tasks(tasks, first_config["task_columns"])

    all_train_files = []
    all_val_files = []
    all_test_files = []

    for dataset_name in dataset_names:
        dataset_files = generate_datalist(
            total_root=total_root,
            dataset_name=dataset_name,
            tasks=normalized_tasks,
        )
        train_files, val_files, test_files = split_by_patient_multitask_balance(
            dataset_files,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
            num_trials=num_trials,
            eval_one_sample_per_patient=eval_one_sample_per_patient,
        )

        all_train_files.extend(train_files)
        all_val_files.extend(val_files)
        all_test_files.extend(test_files)

    return all_train_files, all_val_files, all_test_files

# -----------------------------------------------------------------------------------------------------------------------------------------
# Save and load split data to json files, labels=[0,1,-1] 
# -----------------------------------------------------------------------------------------------------------------------------------------

VALID_SPLITS = {"train", "val", "test"}

def _log(message, logger=None):
    """Log with logger if provided, otherwise print."""
    if logger is not None:
        logger.info(message)
    else:
        print(message)

def _validate_split_sample(sample, split_name, idx):
    """Minimal validation for one split sample."""
    required_keys = ["patient_id", "label"]

    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise KeyError(
            f"Sample {idx} in split '{split_name}' is missing required keys: {missing}"
        )

    if not isinstance(sample["label"], list):
        raise TypeError(
            f"Sample {idx} in split '{split_name}' has invalid label type: "
            f"{type(sample['label'])}. Expected list, e.g. [0] or [0, 1, -1]."
        )

    if "label_mask" in sample:
        if not isinstance(sample["label_mask"], list):
            raise TypeError(
                f"Sample {idx} in split '{split_name}' has invalid label_mask type: "
                f"{type(sample['label_mask'])}. Expected list."
            )

        if len(sample["label"]) != len(sample["label_mask"]):
            raise ValueError(
                f"Sample {idx} in split '{split_name}' has inconsistent label and "
                f"label_mask length: label={sample['label']}, "
                f"label_mask={sample['label_mask']}"
            )

def _check_sample_files_exist(sample, split_name, idx, path_keys=None):
    """Check whether image paths in one sample exist."""
    if path_keys is None:
        path_keys = ["flair", "t1", "t1ce", "t2", "seg"]

    for key in path_keys:
        if key in sample and sample[key] is not None:
            if not os.path.exists(sample[key]):
                raise FileNotFoundError(
                    f"Sample {idx} in split '{split_name}' has missing file "
                    f"for key '{key}': {sample[key]}"
                )

def _count_unique_patients(files):
    return len({sample["patient_id"] for sample in files if "patient_id" in sample})

def _multitask_distribution(samples):
    """Return per-task valid/positive/negative/missing counts."""
    if not samples:
        return {}

    labeled_samples = [sample for sample in samples if "label" in sample]
    if not labeled_samples:
        return {}

    num_tasks = len(labeled_samples[0]["label"])
    stats = {}

    for task_idx in range(num_tasks):
        valid = 0
        positive = 0
        negative = 0
        missing = 0

        for sample in labeled_samples:
            value = sample["label"][task_idx]

            if not _sample_task_is_valid(sample, task_idx):
                missing += 1
            elif value == 1:
                valid += 1
                positive += 1
            elif value == 0:
                valid += 1
                negative += 1
            else:
                raise ValueError(
                    f"Invalid label value {value}. Expected -1, 0, or 1."
                )

        stats[f"task_{task_idx}"] = {
            "valid": valid,
            "positive": positive,
            "negative": negative,
            "missing": missing,
        }

    return stats

def save_data_splits_json(out_dir, train_files, val_files, test_files, logger=None):
    """
    Save train/val/test split files to a JSON file.

    JSON is preferred here because it preserves nested fields such as:
    - label: [0, 1, -1]
    - label_mask: [1, 1, 0]
    - subtype: 2
    - subtype_mask: 1
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "data_splits.json"

    split_data = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, files in split_data.items():
        if files is None:
            raise ValueError(f"{split_name}_files is None")

        for idx, sample in enumerate(files):
            _validate_split_sample(sample, split_name, idx)

    if not any(len(files) > 0 for files in split_data.values()):
        raise ValueError("No samples found. Refusing to save an empty split JSON.")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    for split_name, files in split_data.items():
        _log(
            f"{split_name}: {len(files)} samples, "
            f"{_count_unique_patients(files)} patients",
            logger,
        )
        _log(
            f"{split_name} multi-task distribution: {_multitask_distribution(files)}",
            logger,
        )

    _log(f"Saved data splits to: {json_path}", logger)

    return str(json_path)

def load_data_splits_json(json_path, split=None, check_exists=False, path_keys=None):
    """
    Load train/val/test split files from a JSON file.

    Args:
        json_path: Path to data_splits.json.
        split: None, "train", "val", or "test".
        check_exists: If True, validate that image files exist.
        path_keys: Which keys should be checked as file paths.

    Returns:
        If split is None:
            {"train": train_files, "val": val_files, "test": test_files}
        If split is specified:
            list[dict]
    """
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split: {split}. Expected one of None, 'train', 'val', 'test'."
        )

    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Split JSON does not exist: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    missing_splits = VALID_SPLITS - set(split_data.keys())
    if missing_splits:
        raise KeyError(f"Missing split keys in JSON: {sorted(missing_splits)}")

    invalid_splits = set(split_data.keys()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"JSON contains invalid split keys: {sorted(invalid_splits)}")

    for split_name, files in split_data.items():
        if not isinstance(files, list):
            raise TypeError(
                f"Split '{split_name}' should be a list, got {type(files)}"
            )

        for idx, sample in enumerate(files):
            _validate_split_sample(sample, split_name, idx)

            if check_exists:
                _check_sample_files_exist(sample, split_name, idx, path_keys=path_keys)

    if split is not None:
        return split_data[split]

    return split_data
