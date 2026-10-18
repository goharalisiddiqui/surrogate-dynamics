from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from collective_encoder.datasets.base import BaseDataset


class TimeseriesDataset(Dataset, BaseDataset):
    """
    Dataset for 3-D time-series data of shape ``(n_datapoints, sequence_length, n_features)``.

    Each item is a tuple ``(series, label)`` where:
    - ``series`` is a 2-D tensor of shape ``(sequence_length, n_features)``.
    - ``label`` is a 1-D tensor of shape ``(n_labels,)``.

    Parameters
    ----------
    datapoints : numpy.ndarray or list of numpy.ndarray
        Time-series array of shape ``(n_datapoints, sequence_length, n_features)``
        or list of 2-D arrays of shape ``(sequence_length, n_features)``.
    labels : numpy.ndarray or list of numpy.ndarray
        Corresponding target labels for each datapoint.
    args : dict, optional
        Configuration dictionary forwarded to :class:`BaseDataset`.
    kwargs : dict
        Additional keyword arguments (e.g., ``tag``).
    """

    _IDENTIFIER = "TIMESERIES"
    _REQUIRED_ARGS: List[str] = []
    _OPTIONAL_ARGS: Dict[str, Any] = {}

    def __init__(
        self,
        datapoints: Union[np.ndarray, List[np.ndarray]],
        labels: Union[np.ndarray, List[np.ndarray]],
        args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        Dataset.__init__(self)
        BaseDataset.__init__(self, args=args, **kwargs)

        dtype = torch.get_default_dtype()
        arr = np.asarray(datapoints)
        if arr.ndim == 2:
            # Single feature without trailing feature dim: (n_datapoints, sequence_length) -> (n_datapoints, sequence_length, 1)
            arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3 and len(arr) > 0:
            self.raise_error(
                f"Expected datapoints to have 3 dimensions (n_datapoints, sequence_length, n_features), "
                f"got shape {arr.shape}."
            )

        self.data: List[torch.Tensor] = [
            torch.tensor(arr[i], dtype=dtype) for i in range(len(arr))
        ]
        self.labels: List[torch.Tensor] = [
            torch.tensor(l, dtype=dtype).flatten() for l in labels
        ]

    def __len__(self) -> int:
        """
        Return the number of time-series datapoints in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the ``(series, label)`` tuple at ``index``.
        """
        return self.data[index], self.labels[index]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all time-series datapoints and labels as NumPy arrays.
        """
        return (
            np.array([d.numpy() for d in self.data]),
            np.array([l.numpy() for l in self.labels]),
        )

    def get_norm_data(self) -> np.ndarray:
        """
        Return 2-D array of shape ``(n_datapoints * sequence_length, n_features)``
        for fitting feature-wise normalizers (scalers).
        """
        if len(self.data) == 0:
            return np.empty((0, 0))
        stacked = np.stack([d.numpy() for d in self.data], axis=0)
        return stacked.reshape(-1, stacked.shape[-1])

    def get_datapoint_shape(self) -> tuple:
        """
        Return the shape of a single datapoint ``(sequence_length, n_features)``.
        """
        if len(self.data) == 0:
            return ()
        return tuple(self.data[0].shape)
