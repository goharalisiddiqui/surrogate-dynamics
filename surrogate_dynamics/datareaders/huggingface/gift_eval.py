import os
from typing import Any, Dict, List, Optional

import numpy as np
from huggingface_hub import HfApi

from .base import HuggingfaceReaderBase


class GiftEvalReader(HuggingfaceReaderBase):
    """
    Data reader for the GIFT-Eval benchmark and pretraining datasets hosted on
    Hugging Face (``Salesforce/GiftEvalPretrain`` and ``Salesforce/GiftEval``).

    Uses Hugging Face ``IterableDataset.map(..., batched=True).take(n_datapoints)``
    to stream records from ``sub_dataset``, convert each time series into
    ``(T, n_features)`` layout, and slice long series into non-overlapping chunks
    of ``sequence_length`` steps until at most ``n_datapoints`` (capped at
    ``MAX_DATAPOINTS = 10000``) chunks of shape ``(sequence_length, n_features)``
    are collected.
    """

    _IDENTIFIER = "GIFT-EVAL"
    _DEFAULT_REPO_ID = "Salesforce/GiftEvalPretrain"
    _SUPPORTED_REPOS = (
        "Salesforce/GiftEvalPretrain",
        "Salesforce/GiftEval",
    )

    def _get_repo_id(self) -> str:
        repo = self.repo_id or self._DEFAULT_REPO_ID
        if repo not in self._SUPPORTED_REPOS:
            self.raise_error(
                f"Unsupported repo_id '{repo}' for {self.__class__.__name__}. "
                f"Supported repositories: {list(self._SUPPORTED_REPOS)}"
            )
        return repo

    @classmethod
    def list_available_sub_datasets(
        cls, repo_id: Optional[str] = None
    ) -> List[str]:
        """
        Query the Hugging Face Hub API for available sub-dataset directories in ``repo_id``.
        """
        target_repo = repo_id or cls._DEFAULT_REPO_ID
        api = HfApi()
        entries = api.list_repo_tree(
            target_repo, repo_type="dataset", recursive=True
        )
        sub_datasets = set()
        top_dirs = set()
        for entry in entries:
            path = getattr(entry, "path", "")
            if not path or path.startswith("."):
                continue
            if path.endswith("/dataset_info.json"):
                sub_datasets.add(os.path.dirname(path))
            elif "/" not in path and not path.endswith((".md", ".png", ".json")):
                top_dirs.add(path)
        return sorted(sub_datasets if sub_datasets else top_dirs)

    def _validate_sub_dataset(self) -> None:
        """
        Validate that ``self.sub_dataset`` exists in the configured GIFT-Eval repository.
        If the dataset is already cached locally, offline validation is permitted.
        """
        self.sub_dataset = self.sub_dataset.strip().strip("/")
        repo_id = self._get_repo_id()

        if self.data_dir is not None and os.path.exists(self._get_cache_dirpath()):
            return

        try:
            api = HfApi()
            top_entries = list(
                api.list_repo_tree(repo_id, repo_type="dataset")
            )
            top_paths = {
                e.path
                for e in top_entries
                if not e.path.startswith(".")
                and not e.path.endswith((".md", ".png"))
            }
            top_prefix = self.sub_dataset.split("/")[0]
            if top_prefix not in top_paths:
                self.raise_error(
                    f"Invalid sub_dataset '{self.sub_dataset}' for repository '{repo_id}'. "
                    f"Available top-level sub-datasets: {sorted(top_paths)}"
                )

            if "/" in self.sub_dataset:
                sub_entries = list(
                    api.list_repo_tree(
                        repo_id,
                        path_in_repo=top_prefix,
                        repo_type="dataset",
                        recursive=True,
                    )
                )
                valid_nested = {
                    os.path.dirname(e.path)
                    for e in sub_entries
                    if e.path.endswith("/dataset_info.json")
                } | {e.path for e in sub_entries}
                if self.sub_dataset not in valid_nested:
                    self.raise_error(
                        f"Invalid nested sub_dataset '{self.sub_dataset}' in '{repo_id}'. "
                        f"Available paths under '{top_prefix}': {sorted(valid_nested)}"
                    )
        except ValueError:
            self.raise_error(
                f"Could not verify sub_dataset '{self.sub_dataset}' online via HfApi; "
                "please check your internet connection or the Hugging Face Hub status."
            )
        except Exception as e:
            self.log_warn(
                f"Could not verify sub_dataset '{self.sub_dataset}' online via HfApi ({e}); "
                "proceeding to load/cache check."
            )

    @staticmethod
    def _to_steps_by_features(arr_like: Any) -> np.ndarray:
        """
        Convert a GIFT-Eval time-series array into 2-D shape ``(n_steps, n_features)``.

        In GIFT-Eval:
        - Univariate series have 1-D shape ``(T,)`` -> reshaped to ``(T, 1)``.
        - Multivariate series have 2-D shape ``(n_features, T)`` -> transposed to ``(T, n_features)``.
        """
        arr = np.asarray(arr_like, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        if arr.ndim == 2:
            return arr.T
        raise ValueError(
            f"Expected 1-D or 2-D time-series array in GIFT-Eval record, got shape {arr.shape}."
        )

    def _chunk_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Batched mapping function for ``IterableDataset.map(..., batched=True)``.

        Slices each time-series record in ``batch`` into non-overlapping chunks of
        length ``self.sequence_length``, producing 2-D ``target`` arrays of shape
        ``(sequence_length, n_features)``.
        """
        seq_len = int(self.sequence_length)
        targets = batch.get("target", [])
        covs = batch.get("past_feat_dynamic_real", None)
        item_ids = batch.get("item_id", [""] * len(targets))
        freqs = batch.get("freq", [""] * len(targets))
        starts = batch.get("start", [""] * len(targets))

        out_targets: List[np.ndarray] = []
        out_covs: List[np.ndarray] = []
        out_item_ids: List[str] = []
        out_freqs: List[str] = []
        out_starts: List[str] = []

        include_covs = covs is not None and all(c is not None for c in covs)

        for idx, raw_target in enumerate(targets):
            if raw_target is None:
                continue

            series = self._to_steps_by_features(raw_target)
            n_steps, n_feat = series.shape
            n_chunks = n_steps // seq_len
            if n_chunks == 0:
                continue

            trimmed_len = n_chunks * seq_len
            chunks = series[:trimmed_len].reshape(n_chunks, seq_len, n_feat)

            if include_covs:
                cov_series = self._to_steps_by_features(covs[idx])
                if cov_series.shape[0] >= trimmed_len:
                    cov_chunks = cov_series[:trimmed_len].reshape(
                        n_chunks, seq_len, cov_series.shape[1]
                    )
                    out_covs.extend(list(cov_chunks))
                else:
                    include_covs = False
                    out_covs.clear()

            out_targets.extend(list(chunks))
            out_item_ids.extend([str(item_ids[idx])] * n_chunks)
            out_freqs.extend([str(freqs[idx])] * n_chunks)
            out_starts.extend([str(starts[idx])] * n_chunks)

        result: Dict[str, List[Any]] = {
            "target": out_targets,
            "item_id": out_item_ids,
            "freq": out_freqs,
            "start": out_starts,
        }
        if include_covs and len(out_covs) == len(out_targets):
            result["past_feat_dynamic_real"] = out_covs
        return result