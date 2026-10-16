from typing import Tuple

import numpy as np

from .base import TimeseriesDataModule


class MultivariateTimeseriesDataModule(TimeseriesDataModule):
    """
    PyTorch Lightning DataModule for multivariate time-series data of shape
    ``(n_datapoints, sequence_length, n_features)``.

    Supports arbitrary numbers of feature channels and optional channel subset
    selection via ``feature_indices``.
    """

    _IDENTIFIER = "MULTIVARIATE_TIMESERIES"
    _OPTIONAL_ARGS = TimeseriesDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "feature_indices": None,
    })

    def _process_split_trajs(
        self, trajs: Tuple[np.ndarray, ...]
    ) -> Tuple[np.ndarray, ...]:
        processed = []
        for split_traj in trajs:
            arr = np.asarray(split_traj)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]

            n_features = arr.shape[-1]
            if self.feature_indices is not None:
                if (
                    not isinstance(self.feature_indices, (list, tuple))
                    or len(self.feature_indices) == 0
                ):
                    self.raise_error(
                        "'feature_indices' must be a non-empty list of feature indices."
                    )
                indices = [int(i) for i in self.feature_indices]
                for idx in indices:
                    if idx < -n_features or idx >= n_features:
                        self.raise_error(
                            f"Feature index {idx} in 'feature_indices' is out of bounds "
                            f"for dataset with {n_features} feature(s)."
                        )
                arr = arr[:, :, indices]
            processed.append(arr)
        return tuple(processed)
