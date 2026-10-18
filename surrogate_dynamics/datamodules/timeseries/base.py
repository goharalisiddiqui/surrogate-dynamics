import random
from typing import Any, Dict, List, Tuple

import numpy as np

from collective_encoder.common.registry import get_module
from collective_encoder.datamodules.base import BaseDataModule
from collective_encoder.datasets.resolver import get_dataset_cls_dl


class TimeseriesDataModule(BaseDataModule):
    """
    Base PyTorch Lightning DataModule for 3-D time-series data of shape
    ``(n_datapoints, sequence_length, n_features)``.

    Supports configuring ``sequence_length`` either directly or via
    ``input_chunk_length + output_chunk_length * n_seq_per_sample`` (matching
    :class:`~surrogate_dynamics.datamodules.sequence.SequenceDataModule`), and
    automatically forwards ``sequence_length`` into ``datareader_args``.
    """

    _IDENTIFIER = "TIMESERIES"
    _COMPATIBLE_DATAREADERS = ["GIFT-EVAL", "GIFT-EVAL-PRETRAIN"]
    _COMPATIBLE_DATASETS = ["TIMESERIES"]
    _COMPATIBLE_LABELERS = ["DUMMY", "FROM_EXTERNAL", "CONCAT"]

    _REQUIRED_ARGS = BaseDataModule._REQUIRED_ARGS + [
        "datareader_type",
    ]
    _OPTIONAL_ARGS = BaseDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "dataset_type": "TIMESERIES",
        "datareader_args": None,
        "dataset_args": None,
        "labeler_type": None,
        "labeler_args": None,
        "sequence_length": None,
        "input_chunk_length": None,
        "output_chunk_length": None,
        "n_seq_per_sample": 1,
        "test_full_dataset": False,
        "max_datapoints": None,
        "non_readability_tolerance": 0.02,
        "data_analysers": None,
    })

    def __init__(
        self,
        args: Dict[str, Any] = None,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__(args, **kwargs)

        if self.datareader_args is None:
            self.datareader_args = {}
        else:
            self.datareader_args = self.datareader_args.copy()
        if self.dataset_args is None:
            self.dataset_args = {}
        else:
            self.dataset_args = self.dataset_args.copy()
        if self.labeler_args is None:
            self.labeler_args = {}
        else:
            self.labeler_args = self.labeler_args.copy()

        self._resolve_sequence_length()
        self._initialize_datareader()

        if self.max_datapoints is None:
            self.max_datapoints = self.datareader.get_total_frames()

        if self.test_size > 0 and self.test_full_dataset:
            self.raise_error("test_size and test_full_dataset are mutually exclusive")

        required_frames = (
            self.train_size
            + self.validation_size
            + self.predict_size
            + (0 if self.test_full_dataset else self.test_size)
        )
        if self.max_datapoints < required_frames:
            raise ValueError(
                f"Not enough datapoints ({self.max_datapoints}) in dataset "
                f"for the requested split sizes (train: {self.train_size}, "
                f"val: {self.validation_size}, test: {self.test_size}, "
                f"predict: {self.predict_size})"
            )

        if self.test_full_dataset:
            self.test_size = self.max_datapoints

        self._create_datasets()
        self._analyze_data()

    def _resolve_sequence_length(self) -> None:
        """
        Determine ``self.sequence_length`` from either ``sequence_length`` or
        ``input_chunk_length + output_chunk_length * n_seq_per_sample``, and
        inject it into ``self.datareader_args``.
        """
        has_chunks = (
            self.input_chunk_length is not None
            and self.output_chunk_length is not None
        )
        if has_chunks:
            self.chunk_length = self.input_chunk_length + self.output_chunk_length
            computed_seq_len = (
                self.input_chunk_length
                + self.output_chunk_length * self.n_seq_per_sample
            )
            if (
                self.sequence_length is not None
                and self.sequence_length != computed_seq_len
            ):
                self.raise_error(
                    f"Conflicting sequence_length ({self.sequence_length}) and chunk settings "
                    f"(input_chunk_length={self.input_chunk_length}, "
                    f"output_chunk_length={self.output_chunk_length}, "
                    f"n_seq_per_sample={self.n_seq_per_sample} -> {computed_seq_len})."
                )
            self.sequence_length = computed_seq_len
        elif self.sequence_length is None:
            if "sequence_length" in self.datareader_args:
                self.sequence_length = self.datareader_args["sequence_length"]
            else:
                self.raise_error(
                    "Either 'sequence_length' or both 'input_chunk_length' and "
                    "'output_chunk_length' must be provided."
                )

        self.datareader_args["sequence_length"] = self.sequence_length

    def _initialize_datareader(self) -> None:
        """
        Instantiate the configured time-series datareader.
        """
        datareader_cls = get_module("datareaders", self.datareader_type)
        args = self.datareader_args.copy()
        args['n_datapoints'] = self.max_datapoints
        self.datareader = datareader_cls(
            args=args,
            **self.get_run_args()
        )
        self.log_info(
            f"Total datapoints in datareader: {self.datareader.get_total_frames()}"
        )

    def _validate_traj_length(
        self,
        traj: Any,
        expected_length: int,
        failed_indices: List[int] = None,
        split_name: str = "",
    ) -> None:
        """
        Validate that the read split has the expected number of datapoints.
        """
        if failed_indices:
            self.warn(
                f"[{split_name}] {len(failed_indices)} datapoint(s) could not be read "
                f"and were skipped. Failed indices: {failed_indices}"
            )
        threshold = expected_length * (1 - self.non_readability_tolerance)
        if len(traj) < threshold:
            raise ValueError(
                f"[{split_name}] Split length {len(traj)} is less than the minimum expected "
                f"{threshold:.0f} datapoints (tolerance={self.non_readability_tolerance:.1%}). "
                f"Failed indices: {failed_indices}"
            )

    def _calculate_indices(self) -> None:
        """
        Calculate train, validation, test, and predict indices based on split configuration.
        """
        if self.max_datapoints > self.datareader.get_total_frames():
            raise ValueError(
                f"max_frames ({self.max_datapoints}) exceeds total datapoints in datareader "
                f"({self.datareader.get_total_frames()})"
            )
        self.log_debug("Calculating split indices...")
        required_size = self.train_size + self.validation_size + self.predict_size
        self.log_debug(f"Base required size (train + val + predict): {required_size}")
        if not self.test_full_dataset:
            required_size += self.test_size
        self.log_debug(f"Total required size (including test): {required_size}")

        start = random.randint(0, self.max_datapoints - required_size)
        self.train_indices = np.arange(start, start + self.train_size)
        if len(self.train_indices) > 0:
            self.log_info(
                f"Train indices: {self.train_indices[0]} - {self.train_indices[-1]}"
            )

        val_start = start + self.train_size
        val_end = val_start + self.validation_size
        self.val_indices = np.arange(val_start, val_end)
        if len(self.val_indices) > 0:
            self.log_info(
                f"Validation indices: {self.val_indices[0]} - {self.val_indices[-1]}"
            )

        if self.test_full_dataset:
            self.test_indices = np.arange(self.max_datapoints)
            pred_start = val_end
        else:
            test_end = val_end + self.test_size
            self.test_indices = np.arange(val_end, test_end)
            pred_start = test_end
        if len(self.test_indices) > 0:
            self.log_info(
                f"Test indices: {self.test_indices[0]} - {self.test_indices[-1]}"
            )

        self.predict_indices = np.arange(pred_start, pred_start + self.predict_size)
        if len(self.predict_indices) > 0:
            self.log_info(
                f"Predict indices: {self.predict_indices[0]} - {self.predict_indices[-1]}"
            )

    def _process_split_trajs(
        self, trajs: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, ...]:
        """
        Hook for subclasses to validate or slice the feature axis (3rd dimension)
        of each split array in ``trajs``.
        """
        return trajs

    def _create_datasets(self) -> None:
        """
        Create train, validation, test, and predict datasets.
        """
        self.log_info("Calculating dataset split indices...")
        self._calculate_indices()
        self.log_info("Dataset split indices calculated.")

        self.log_msg(
            f"Train size: {self.train_size}, "
            f"Validation size: {self.validation_size}, "
            f"Test size: {self.test_size}, "
            f"Predict size: {self.predict_size}"
        )

        self.log_info("Reading time-series data and labels...")
        read_result = self.datareader.read_trajectory(
            indices=[
                self.train_indices,
                self.val_indices,
                self.test_indices,
                self.predict_indices,
            ],
            labeler_type=self.labeler_type,
            labeler_args=self.labeler_args,
        )
        self.log_info("Finished reading time-series data and labels.")

        if len(read_result) == 3:
            trajs, labels, failed_per_split = read_result
        else:
            trajs, labels = read_result
            failed_per_split = ([], [], [], [])

        trajs = self._process_split_trajs(tuple(trajs))

        self._validate_traj_length(
            trajs[0], len(self.train_indices), failed_per_split[0], split_name="train"
        )
        self._validate_traj_length(
            trajs[1], len(self.val_indices), failed_per_split[1], split_name="val"
        )
        self._validate_traj_length(
            trajs[2], len(self.test_indices), failed_per_split[2], split_name="test"
        )
        self._validate_traj_length(
            trajs[3],
            len(self.predict_indices),
            failed_per_split[3],
            split_name="predict",
        )

        dataset_class, dataset_args, self.dl_cls = get_dataset_cls_dl(
            self.dataset_type,
            self.dataset_args,
            self.datareader,
        )

        if self.train_size > 0:
            self.log_info(
                f"Creating training dataset with {len(trajs[0])} datapoints..."
            )
            self.train_data = dataset_class(
                datapoints=trajs[0],
                labels=labels[0],
                args=dataset_args,
                **{**self.run_args, "tag": "train"},
            )
        else:
            self.train_data = []

        if self.validation_size > 0:
            self.log_info(
                f"Creating validation dataset with {len(trajs[1])} datapoints..."
            )
            self.val_data = dataset_class(
                datapoints=trajs[1],
                labels=labels[1],
                args=dataset_args,
                **{**self.run_args, "verbose": False, "tag": "val"},
            )
        else:
            self.val_data = []

        if self.test_size > 0:
            self.log_info(f"Creating test dataset with {len(trajs[2])} datapoints...")
            self.test_data = dataset_class(
                datapoints=trajs[2],
                labels=labels[2],
                args=dataset_args,
                **{**self.run_args, "verbose": False, "tag": "test"},
            )
        else:
            self.test_data = []

        if self.predict_size > 0:
            self.log_info(
                f"Creating predict dataset with {len(trajs[3])} datapoints..."
            )
            self.predict_data = dataset_class(
                datapoints=trajs[3],
                labels=labels[3],
                args=dataset_args,
                **{**self.run_args, "verbose": False, "tag": "predict"},
            )
        else:
            self.predict_data = []

        self.num_frames = (
            len(self.train_data)
            + len(self.val_data)
            + len(self.test_data)
            + len(self.predict_data)
        )
        active_dataset = self.get_any_dataset()
        self.datapoint_shape = active_dataset.get_datapoint_shape()
        self.label_list = self.datareader.get_label_names()

        self.log_msg(
            f"Loaded dataset with {self.num_frames} datapoints -> "
            f"Train: {len(self.train_data)}, "
            f"Validation: {len(self.val_data)}, "
            f"Test: {len(self.test_data)}, "
            f"Predict: {len(self.predict_data)}"
        )
        self.log_msg(f"Datapoint shape: {self.datapoint_shape}")

        self._check_batch_sizes()
