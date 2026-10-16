import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset, IterableDataset, load_dataset, load_from_disk

from collective_encoder.common.registry import get_module
from collective_encoder.datalabelers.base import BaseLabeler
from collective_encoder.datareaders.base import BaseDataReader
from collective_encoder.datareaders.processors.resolver import get_processor


class HuggingfaceReaderBase(BaseDataReader, ABC):
    """
    Abstract base class for streaming, chunking, caching, and reading datasets
    from Hugging Face Hub using the Hugging Face ``datasets`` Python API.

    Uses native Hugging Face ``IterableDataset`` streaming pipeline:
    1. ``load_dataset(..., streaming=True)`` to stream shards lazily without
       downloading the full repository.
    2. ``stream.map(self._chunk_batch, batched=True, batch_size=1)`` to slice
       variable-length records into fixed ``sequence_length`` datapoints of
       shape ``(sequence_length, n_features)``.
    3. ``stream.take(self.n_datapoints)`` to cap the number of downloaded
       datapoints (hard-limited by ``MAX_DATAPOINTS = 10000``).
    4. ``Dataset.save_to_disk()`` / ``load_from_disk().with_format("numpy")``
       to cache and reload the resulting Arrow dataset locally.

    Subclasses must implement:
    - :meth:`_get_repo_id`
    - :meth:`_validate_sub_dataset`
    - :meth:`_chunk_batch`
    """

    MAX_DATAPOINTS: int = 10000

    _COMPATIBLE_LABELERS = BaseDataReader._COMPATIBLE_LABELERS + [
        "FROM_EXTERNAL",
    ]
    _IDENTIFIER: Optional[str] = None  # To be defined by subclasses
    _REQUIRED_ARGS = BaseDataReader._REQUIRED_ARGS + [
        "sub_dataset",
        "sequence_length",
    ]
    _OPTIONAL_ARGS = BaseDataReader._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "n_datapoints": MAX_DATAPOINTS,
        "data_dir": None,
        "repo_id": None,
        "split": "train",
        "force_download": False,
        "processors": None,
    })

    def __init__(self, args: Dict[str, Any] = None, **kwargs):
        super().__init__(args=args, **kwargs)

        self.processors = []
        processor_configs = self.args.get("processors") or []
        for config in processor_configs:
            processor_cls = get_processor(config["type"])
            processor_args = config.get("args", {})
            self.processors.append(processor_cls(**processor_args))
            self.log_msg(
                f"Initialized processor: {config['type']} with args {processor_args}"
            )

        self._validate_arguments()
        self._validate_sub_dataset()
        self._setup_data_dir()
        self._download_and_load()

    def _validate_arguments(self) -> None:
        """
        Validate ``n_datapoints`` (against ``MAX_DATAPOINTS``) and ``sequence_length``.
        """
        if self.n_datapoints is None:
            self.n_datapoints = self.MAX_DATAPOINTS

        if not isinstance(self.n_datapoints, (int, np.integer)) or self.n_datapoints <= 0:
            self.raise_error(
                f"'n_datapoints' must be a positive integer, got {self.n_datapoints!r}."
            )
        if self.n_datapoints > self.MAX_DATAPOINTS:
            self.raise_error(
                f"Requested n_datapoints ({self.n_datapoints}) exceeds the maximum "
                f"allowed limit of {self.MAX_DATAPOINTS}."
            )

        if (
            not isinstance(self.sequence_length, (int, np.integer))
            or self.sequence_length <= 0
        ):
            self.raise_error(
                f"'sequence_length' must be a positive integer, got {self.sequence_length!r}."
            )

        if not isinstance(self.sub_dataset, str) or not self.sub_dataset.strip():
            self.raise_error("'sub_dataset' must be a non-empty string.")

    @abstractmethod
    def _get_repo_id(self) -> str:
        """
        Return the Hugging Face dataset repository ID (e.g., ``'Salesforce/GiftEvalPretrain'``).
        """
        raise NotImplementedError("Subclasses must implement _get_repo_id")

    @abstractmethod
    def _validate_sub_dataset(self) -> None:
        """
        Validate that ``self.sub_dataset`` is a valid sub-dataset in the target repository.
        """
        raise NotImplementedError("Subclasses must implement _validate_sub_dataset")

    @abstractmethod
    def _chunk_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Batched mapping function passed to ``IterableDataset.map(..., batched=True)``.

        Takes a batch of raw records from Hugging Face and returns a dictionary
        of lists where ``'target'`` contains 2-D arrays of shape
        ``(sequence_length, n_features)``.
        """
        raise NotImplementedError("Subclasses must implement _chunk_batch")

    def _get_default_data_dir_name(self) -> str:
        """
        Default directory name created when ``data_dir`` is not explicitly provided.
        """
        ident = (self._IDENTIFIER or "huggingface").lower().replace("-", "_")
        return f"{ident}_data"

    def _setup_data_dir(self) -> None:
        """
        Set up the local directory for caching downloaded/sliced dataset files.
        """
        if self.data_dir is not None:
            os.makedirs(self.data_dir, exist_ok=True)
            self.log_info(f"Using specified data directory: {self.data_dir}")
        else:
            self._create_dir(
                self._get_default_data_dir_name(),
                var_name="data_dir",
                use_existing=True,
                in_exec_dir=True,
            )

    def _get_cache_dirpath(self) -> str:
        """
        Return the local directory path used by ``Dataset.save_to_disk`` / ``load_from_disk``.
        """
        repo_slug = self._get_repo_id().replace("/", "__")
        sub_slug = self.sub_dataset.strip("/").replace("/", "__")
        dirname = (
            f"{repo_slug}__{sub_slug}__{self.split}"
            f"__seq{self.sequence_length}__n{self.n_datapoints}"
        )
        return os.path.join(self.data_dir, dirname)

    def _stream_dataset(self) -> IterableDataset:
        """
        Open a streaming ``IterableDataset`` from Hugging Face Hub.
        Supports both standard Hub dataset configs and Arrow ``save_to_disk``
        subdirectories via the ``hf://`` protocol.
        """
        repo_id = self._get_repo_id()
        sub = self.sub_dataset.strip("/")
        self.log_msg(
            f"Streaming sub-dataset '{sub}' (split='{self.split}') from "
            f"Hugging Face repo '{repo_id}' (taking up to {self.n_datapoints} "
            f"datapoints of sequence_length={self.sequence_length})..."
        )
        try:
            return load_dataset(
                repo_id,
                data_dir=sub,
                split=self.split,
                streaming=True,
            )
        except Exception:
            try:
                arrow_pattern = f"hf://datasets/{repo_id}/{sub}/*.arrow"
                return load_dataset(
                    "arrow",
                    data_files={self.split: arrow_pattern},
                    split=self.split,
                    streaming=True,
                )
            except Exception as e:
                self.raise_error(
                    f"Failed to stream sub-dataset '{sub}' from '{repo_id}': {e}"
                )

    def _download_and_load(self) -> None:
        """
        Load cached Hugging Face ``Dataset`` from disk if available; otherwise
        stream using ``IterableDataset.map(...).take(n_datapoints)`` and save
        to disk via ``Dataset.save_to_disk()``.
        """
        cache_dir = self._get_cache_dirpath()

        if os.path.exists(cache_dir) and not self.force_download:
            self.log_msg(f"Loading cached Hugging Face dataset from {cache_dir}")
            try:
                hf_ds = load_from_disk(cache_dir).with_format("numpy")
            except Exception as e:
                self.raise_error(
                    f"Failed to load cached dataset from '{cache_dir}': {e}"
                )
        else:
            stream = self._stream_dataset()
            if not isinstance(stream, IterableDataset):
                stream = IterableDataset.from_generator(lambda: (row for row in stream))

            max_to_take = min(int(self.n_datapoints), self.MAX_DATAPOINTS)
            chunked_stream = stream.map(
                self._chunk_batch,
                batched=True,
                batch_size=1,
                remove_columns=stream.column_names,
            ).take(max_to_take)

            records = list(chunked_stream)
            if len(records) == 0:
                self.raise_error(
                    f"No time-series chunks of sequence_length={self.sequence_length} "
                    f"could be extracted from sub-dataset '{self.sub_dataset}'."
                )

            hf_ds = Dataset.from_list(records).with_format("numpy")
            try:
                hf_ds.save_to_disk(cache_dir)
                self.log_msg(f"Saved chunked Hugging Face dataset to {cache_dir}")
            except Exception as e:
                self.log_warn(f"Could not save cache to {cache_dir}: {e}")

        if len(hf_ds) > self.MAX_DATAPOINTS:
            hf_ds = hf_ds.select(range(self.MAX_DATAPOINTS))

        self.hf_dataset = hf_ds
        self.raw_data = {col: np.asarray(hf_ds[col]) for col in hf_ds.column_names}
        self.data = self.raw_data["target"]

        if self.data.ndim != 3:
            self.raise_error(
                f"Expected 3-D data array of shape (n_datapoints, sequence_length, n_features), "
                f"got shape {self.data.shape}."
            )

        self.n_frames, self.n_steps, self.n_features = self.data.shape
        self.log_msg(
            f"Loaded dataset '{self.sub_dataset}' with shape {self.data.shape} "
            f"(datapoints={self.n_frames}, steps={self.n_steps}, features={self.n_features})."
        )

    def get_total_frames(self) -> int:
        """
        Return the total number of datapoints (frames) available in the dataset.
        """
        total = self.n_frames
        if hasattr(self, "processors"):
            for processor in self.processors:
                total = processor.adjust_total_frames(total)
        return total

    def _initialize_labeler(
        self, labeler_type: str, labeler_args: dict
    ) -> BaseLabeler:
        """
        Instantiate the labeler for this Hugging Face dataset reader, passing ``self.raw_data``.
        """
        labeler_cls = get_module("datalabelers", labeler_type)
        run_args = self.get_run_args().copy()
        run_args.pop("dataset", None)
        return labeler_cls(
            dataset=self.raw_data,
            args=labeler_args,
            **run_args,
        )

    def read(
        self,
        labeler_type: str = None,
        labeler_args: dict = None,
        **kwargs,
    ):
        """
        Read the trajectory data and compute labels for the given indices.
        """
        return self.read_trajectory(
            indices=kwargs.get("indices", []),
            labeler_type=labeler_type,
            labeler_args=labeler_args or {},
        )

    def read_trajectory(
        self,
        indices: List[Union[List[int], np.ndarray]],
        labeler_type: Optional[str] = "DUMMY",
        labeler_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[List[int], ...]]:
        """
        Read requested datapoint splits and compute corresponding labels.
        """
        if labeler_args is None:
            labeler_args = {}

        labeler = self.initialize_labeler(
            labeler_type=labeler_type,
            labeler_args=labeler_args,
        )
        self.label_list = labeler.get_label_names()

        self.log_msg("Reading time-series datapoints and computing labels...")

        prepared_indices = []
        for seq in indices:
            processed_seq = list(seq)
            for processor in self.processors:
                processed_seq = processor.prepare_seq(processed_seq)
            prepared_indices.append(processed_seq)

        trajs, labels, all_failed = (), (), ()
        for seq_indices in prepared_indices:
            if len(seq_indices) == 0:
                traj_split = np.empty(
                    (0, self.n_steps, self.n_features), dtype=self.data.dtype
                )
                label_split = np.empty((0, len(self.label_list)), dtype=np.float32)
                failed_split: List[int] = []
            else:
                valid_indices = []
                failed_split = []
                for idx in seq_indices:
                    if 0 <= int(idx) < self.n_frames:
                        valid_indices.append(int(idx))
                    else:
                        failed_split.append(int(idx))

                if len(valid_indices) > 0:
                    traj_split = self.data[valid_indices]
                    label_split = np.asarray(labeler.compute(valid_indices))
                else:
                    traj_split = np.empty(
                        (0, self.n_steps, self.n_features), dtype=self.data.dtype
                    )
                    label_split = np.empty(
                        (0, len(self.label_list)), dtype=np.float32
                    )

            for processor in self.processors:
                traj_split, label_split = processor.postprocess_seq(
                    traj_split, label_split
                )

            trajs += (traj_split,)
            labels += (label_split,)
            all_failed += (failed_split,)

        self.log_msg(
            f"Finished reading Hugging Face dataset. "
            f"Split shapes: {[t.shape for t in trajs]}, "
            f"label shapes: {[l.shape for l in labels]}."
        )
        return trajs, labels, all_failed
