from typing import Tuple

import numpy as np

from .base import TimeseriesDataModule


class UnivariateTimeseriesDataModule(TimeseriesDataModule):
    """
    PyTorch Lightning DataModule for univariate time-series data of shape
    ``(n_datapoints, sequence_length, 1)``.

    Requires the underlying time-series data to have ``n_features == 1``, or
    selects a single feature channel when ``feature_index`` is specified.
    """

    _IDENTIFIER = "UNIVARIATE_TIMESERIES"
    _OPTIONAL_ARGS = TimeseriesDataModule._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "feature_index": None,
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
            if self.feature_index is not None:
                idx = int(self.feature_index)
                if idx < -n_features or idx >= n_features:
                    self.raise_error(
                        f"feature_index ({self.feature_index}) is out of bounds for "
                        f"dataset with {n_features} feature(s)."
                    )
                arr = arr[:, :, [idx]]
            elif n_features != 1:
                self.raise_error(
                    f"{self.__class__.__name__} requires univariate data (n_features == 1) "
                    f"or 'feature_index' to be specified, but loaded data has {n_features} features."
                )
            processed.append(arr)
        return tuple(processed)
