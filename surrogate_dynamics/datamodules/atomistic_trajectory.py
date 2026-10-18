import warnings
from typing import Any, Dict, List

import torch
from collective_encoder.datamodules.coordinates import CoordinatesDataModule

warnings.filterwarnings("ignore")


class AtomisticTrajectoryDataModule(CoordinatesDataModule):
    """
    DataModule for atomistic trajectory sequence prediction tasks.

    Loads sequences as chunks of consecutive frames from a molecular trajectory
    by injecting a ``Chunk`` processor into the underlying trajectory datareader
    (e.g., ``COMPRESSED_TRAJECTORY``), and scales dataloader batch sizes by
    ``sequence_length`` so each batch contains ``batch_size`` complete sequences
    of consecutive frames.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary. Supports all :class:`CoordinatesDataModule`
        arguments plus:
        - ``sequence_length`` (int, optional): Total number of frames per sequence.
        - ``input_chunk_length`` (int, optional): Number of frames used as input.
        - ``output_chunk_length`` (int, optional): Number of frames used as output.
        - ``n_seq_per_sample`` (int, default 1): Number of output chunks per sequence.
          When ``input_chunk_length`` and ``output_chunk_length`` are given,
          ``sequence_length = input_chunk_length + output_chunk_length * n_seq_per_sample``.
    kwargs : dict
        Additional keyword arguments forwarded to :class:`CoordinatesDataModule`.
    """

    _IDENTIFIER = "ATOMISTIC_TRAJECTORY"
    _COMPATIBLE_DATAREADERS = CoordinatesDataModule._COMPATIBLE_DATAREADERS
    _COMPATIBLE_DATASETS = CoordinatesDataModule._COMPATIBLE_DATASETS
    _COMPATIBLE_LABELERS = CoordinatesDataModule._COMPATIBLE_LABELERS
    _REQUIRED_ARGS = CoordinatesDataModule._REQUIRED_ARGS.copy()
    _OPTIONAL_ARGS = CoordinatesDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "sequence_length": None,
        "input_chunk_length": None,
        "output_chunk_length": None,
        "n_seq_per_sample": 1,
    })

    def _resolve_sequence_length(self) -> None:
        """
        Determine ``self.sequence_length`` from either ``sequence_length`` or
        ``input_chunk_length + output_chunk_length * n_seq_per_sample``.
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
            self.raise_error(
                "Either 'sequence_length' or both 'input_chunk_length' and "
                "'output_chunk_length' must be provided."
            )
        else:
            self.chunk_length = self.sequence_length

    def _inject_chunk_processor(self) -> None:
        """
        Ensure ``self.datareader_args['processors']`` includes a sequence-expanding
        processor configured with ``self.sequence_length``.
        """
        self.datareader_args = (self.datareader_args or {}).copy()
        raw_processors = self.datareader_args.get("processors") or []
        processors: List[Dict[str, Any]] = [
            {
                "type": p["type"],
                "args": dict(p.get("args") or {}),
            }
            for p in raw_processors
        ]

        injected = False
        for proc in processors:
            if proc["type"] in ("Chunk", "CoarseGrain"):
                existing_len = proc["args"].get("sequence_length")
                if existing_len is not None and existing_len != self.sequence_length:
                    self.raise_error(
                        f"Processor '{proc['type']}' has sequence_length={existing_len}, "
                        f"which conflicts with DataModule sequence_length={self.sequence_length}."
                    )
                proc["args"]["sequence_length"] = self.sequence_length
                injected = True
                break

        if not injected:
            processors.insert(
                0,
                {
                    "type": "Chunk",
                    "args": {"sequence_length": self.sequence_length},
                },
            )

        self.datareader_args["processors"] = processors

    def _initialize_datareader(self) -> None:
        """
        Resolve ``sequence_length``, inject the ``Chunk`` processor into
        ``datareader_args['processors']``, and initialize the trajectory reader.
        """
        self._resolve_sequence_length()
        self._inject_chunk_processor()
        super()._initialize_datareader()

    def _validate_traj_length(
        self,
        traj: List[Any],
        expected_length: int,
        failed_indices: List[int] = None,
        split_name: str = "",
    ) -> None:
        """
        Validate that the read trajectory split contains ``expected_length * sequence_length`` frames.
        """
        super()._validate_traj_length(
            traj=traj,
            expected_length=expected_length * self.sequence_length,
            failed_indices=failed_indices,
            split_name=split_name,
        )

    def train_dataloader(self):
        """Return training dataloader with ``batch_size * sequence_length`` frames per batch."""
        return self.get_dataloader(
            self.train_data,
            self.batch_size * self.sequence_length,
            shuffle=False,
        )

    def val_dataloader(self):
        """Return validation dataloader with ``val_batch_size * sequence_length`` frames per batch."""
        if self.val_data is None or len(self.val_data) == 0:
            raise ValueError("Validation data is not available")
        return self.get_dataloader(
            self.val_data,
            self.val_batch_size * self.sequence_length,
            shuffle=False,
        )

    def test_dataloader(self):
        """Return test dataloader with ``test_batch_size * sequence_length`` frames per batch."""
        if self.test_data is None or len(self.test_data) == 0:
            raise ValueError("Test data is not available")
        return self.get_dataloader(
            self.test_data,
            self.test_batch_size * self.sequence_length,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Return prediction dataloader."""
        if self.predict_data is None or len(self.predict_data) == 0:
            raise ValueError("Prediction data is not available")
        batch_frames = (
            self.predict_batch_size * self.sequence_length
            if self.predict_batch_size > 0
            else len(self.predict_data)
        )
        return self.get_dataloader(
            self.predict_data,
            min(batch_frames, len(self.predict_data)),
            shuffle=False,
        )
