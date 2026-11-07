from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from torch.utils.data import DataLoader, Dataset

from collective_encoder.datareaders.base import BaseDataReader
from collective_encoder.datasets.base import BaseDataset
from surrogate_dynamics.propagators.moirai2.modules.imputation import CausalMeanImputation


class DartsTimeseriesDataset(Dataset, BaseDataset):
    """
    Dataset backed by :class:`darts.TimeSeries` that inherits from
    :class:`~collective_encoder.datasets.base.BaseDataset`.

    Converts each time-series datapoint of shape ``(sequence_length, n_features)``
    into a :class:`darts.TimeSeries` object, applies optional Darts time-series
    transformations and covariate builders (such as :meth:`TimeSeries.diff`,
    :meth:`TimeSeries.window_transform`, :meth:`TimeSeries.add_datetime_attribute`,
    :meth:`TimeSeries.add_holidays`, relative time index, and component splitting),
    and splits each series into context (``context_length``)
    and prediction horizon (``prediction_length``) windows.

    Each item returned by :meth:`__getitem__` is a tuple ``(sample_dict, label)`` where:
    - ``sample_dict`` is a dictionary of PyTorch tensors ready for forecasting models:
        * ``"past_target"``: ``FloatTensor`` of shape ``(context_length, target_dim)`` the past target values
        * ``"past_observed_target"``: ``BoolTensor`` of shape ``(context_length, target_dim)`` 
          indicating observed (non-NaN) values
        * ``"past_is_pad"``: ``BoolTensor`` of shape ``(context_length,)``
          indicating whether each past time step is padding (True) or real data (False)
        * ``"future_target"``: ``FloatTensor`` of shape ``(prediction_length, target_dim)``
          (included when ``prediction_length > 0``)
          the future target values (ground truth for forecasting)
        * ``"future_observed_target"``: ``BoolTensor`` of shape ``(prediction_length, target_dim)``
          (included when ``prediction_length > 0``)
          indicating observed (non-NaN) values in the future target
        * ``"future_is_pad"``: ``BoolTensor`` of shape ``(prediction_length,)``
          (included when ``prediction_length > 0``)
          indicating whether each future time step is padding (True) or real data (False)
        * ``"feat_dynamic_real"``: ``FloatTensor`` of shape
          ``(context_length + prediction_length, feat_dynamic_real_dim)`` (when present)
          the dynamic covariate values for the full window
        * ``"observed_feat_dynamic_real"``: ``BoolTensor`` of shape
          ``(context_length + prediction_length, feat_dynamic_real_dim)`` (when present)
            indicating observed (non-NaN) values for the dynamic covariates
        * ``"past_feat_dynamic_real"``: ``FloatTensor`` of shape
          ``(context_length, past_feat_dynamic_real_dim)`` (when present)
          the past dynamic covariate values for the context window
        * ``"past_observed_feat_dynamic_real"``: ``BoolTensor`` of shape
          ``(context_length, past_feat_dynamic_real_dim)`` (when present)
          indicating observed (non-NaN) values for the past dynamic covariates
        * ``"static_covariates"``: ``FloatTensor`` (when static covariates are present)
          the static covariate values for the sample, flattened to 1-D
          
    - ``label`` is a 1-D ``FloatTensor`` of shape ``(n_labels,)``.
    """

    _IDENTIFIER = "DARTS_TIMESERIES"
    _REQUIRED_ARGS: List[str] = [
        "context_length",
        "prediction_length",
    ]
    _OPTIONAL_ARGS: Dict[str, Any] = {
        "target_components": None,
        "feat_dynamic_real_components": None,
        "past_feat_dynamic_real_components": None,
        "static_covariates": None,
        "freq": None,
        "start": None,
        "diff_n": None,
        "window_transforms": None,
        "datetime_attributes": None,
        "country_holidays": None,
        "add_relative_index": False,
        "impute_missing": True,
        "fillna_value": 0.0,
        "sequence_length": None,
    }

    @classmethod
    def _prepare_dataset(
        cls,
        dataset_args: Dict[str, Any],
        datareader: BaseDataReader,
    ) -> Tuple[type, Dict[str, Any], type]:
        """
        Prepare dataset arguments using metadata from ``datareader`` when available.
        """
        prepared_args = (dataset_args or {}).copy()
        if datareader is not None:
            if prepared_args.get("sequence_length") is None and hasattr(
                datareader, "sequence_length"
            ):
                prepared_args["sequence_length"] = datareader.sequence_length

            raw_data = getattr(datareader, "raw_data", None)
            if isinstance(raw_data, dict):
                if prepared_args.get("freq") is None and "freq" in raw_data:
                    freqs = raw_data["freq"]
                    if len(freqs) > 0 and freqs[0]:
                        prepared_args["freq"] = str(freqs[0])
                if prepared_args.get("start") is None and "start" in raw_data:
                    starts = raw_data["start"]
                    if len(starts) > 0 and starts[0]:
                        prepared_args["start"] = str(starts[0])
        return cls, prepared_args, DataLoader

    def __init__(
        self,
        datapoints: Union[np.ndarray, Sequence[Union[np.ndarray, TimeSeries]]],
        labels: Union[np.ndarray, Sequence[np.ndarray]],
        args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Receives a list of time-series 
        """
        Dataset.__init__(self)
        BaseDataset.__init__(self, args=args, **kwargs)

        self._imputer = CausalMeanImputation(value=float(self.fillna_value or 0.0))

        raw_series_list = self._identify_input_series(datapoints)
        self.labels: List[torch.Tensor] = [
            torch.tensor(lbl, dtype=self.dtype).flatten() for lbl in labels
        ]

        # Build Darts TimeSeries instances and extract target/covariate TimeSeries
        self.timeseries_list: List[TimeSeries] = []
        self._target_ts_list: List[TimeSeries] = []
        self._future_cov_ts_list: List[Optional[TimeSeries]] = []
        self._past_cov_ts_list: List[Optional[TimeSeries]] = []

        for idx, item in enumerate(raw_series_list):
            base_ts = (
                item.copy()
                if isinstance(item, TimeSeries)
                else self._create_darts_timeseries(item, sample_idx=idx)
            )
            target_ts, fut_cov_ts, past_cov_ts = self._apply_darts_operations(base_ts)
            self.timeseries_list.append(base_ts)
            self._target_ts_list.append(target_ts)
            self._future_cov_ts_list.append(fut_cov_ts)
            self._past_cov_ts_list.append(past_cov_ts)
            

        if self.context_length < 0 or self.prediction_length < 0:
            self.raise_error(
                f"context_length ({self.context_length}) and prediction_length "
                f"({self.prediction_length}) must be non-negative."
            )
        self.total_window_length: int = self.context_length + self.prediction_length
        if self.sequence_length is not None and self.sequence_length != self.total_window_length:
            self.raise_error(
                f"sequence_length ({self.sequence_length}) must equal "
                f"context_length + prediction_length ({self.total_window_length})."
            )

        self.target_dim: int = (
            int(self._target_ts_list[0].width) if len(self._target_ts_list) > 0 else 0
        )
        self.feat_dynamic_real_dim: int = (
            int(self._future_cov_ts_list[0].width)
            if len(self._future_cov_ts_list) > 0
            and self._future_cov_ts_list[0] is not None
            else 0
        )
        self.past_feat_dynamic_real_dim: int = (
            int(self._past_cov_ts_list[0].width)
            if len(self._past_cov_ts_list) > 0
            and self._past_cov_ts_list[0] is not None
            else 0
        )

        self.samples: List[Dict[str, torch.Tensor]] = [
            self._build_sample_dict(i) for i in range(len(self._target_ts_list))
        ]

    # ------------------------------------------------------------------
    # Internal Helpers: Darts TimeSeries Construction & Transforms
    # ------------------------------------------------------------------

    def _identify_input_series(
        self, datapoints: Union[np.ndarray, Sequence[Union[np.ndarray, TimeSeries]]]
    ) -> List[Union[np.ndarray, TimeSeries]]:
        """
        Identify the type of datapoints and return either a list of 
        :class:`darts.TimeSeries` or a list of 2-D NumPy arrays.
        """
        if isinstance(datapoints, (list, tuple)) and len(datapoints) > 0 and isinstance(
            datapoints[0], TimeSeries
        ):
            return list(datapoints)

        if isinstance(datapoints, (list, tuple)):
            if len(datapoints) == 0:
                return []
            out: List[np.ndarray] = []
            for item in datapoints:
                arr = np.asarray(item, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                elif arr.ndim != 2:
                    self.raise_error(
                        f"Expected each time-series item to be 1-D or 2-D, got shape {arr.shape}."
                    )
                out.append(arr)
            return out

        arr = np.asarray(datapoints, dtype=np.float32)
        if len(arr) == 0:
            return []
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3:
            self.raise_error(
                f"Expected datapoints to have 3 dimensions (n_datapoints, sequence_length, n_features), "
                f"got shape {arr.shape}."
            )
        return [arr[i] for i in range(len(arr))]

    def _create_darts_timeseries(
        self, values: np.ndarray, sample_idx: int = 0
    ) -> TimeSeries:
        """
        Construct a :class:`darts.TimeSeries` from a 2-D NumPy array ``(n_steps, n_features)``.
        Uses :meth:`TimeSeries.from_times_and_values` when ``freq`` is configured,
        falling back to :meth:`TimeSeries.from_values` (RangeIndex) otherwise.
        """
        n_steps, n_features = values.shape
        columns = [f"feature_{j}" for j in range(n_features)]

        static_cov_df = None
        if self.static_covariates is not None:
            sc = np.asarray(self.static_covariates, dtype=np.float32)
            if sc.ndim == 1:
                static_cov_df = pd.DataFrame([sc])
            elif sc.ndim == 2 and sample_idx < len(sc):
                static_cov_df = pd.DataFrame([sc[sample_idx]])

        if self.freq is not None:
            start_ts = self.start if self.start is not None else "2000-01-01"
            try:
                time_index = pd.date_range(
                    start=start_ts, periods=n_steps, freq=str(self.freq)
                )
                return TimeSeries.from_times_and_values(
                    times=time_index,
                    values=values,
                    columns=columns,
                    static_covariates=static_cov_df,
                    fill_missing_dates=False,
                )
            except Exception as e:
                self.log_warn(
                    f"Could not construct DatetimeIndex with freq={self.freq!r}, "
                    f"start={start_ts!r} ({e}); falling back to RangeIndex."
                )

        return TimeSeries.from_values(
            values=values,
            columns=columns,
            static_covariates=static_cov_df,
        )

    @staticmethod
    def _select_components(
        ts: TimeSeries, selector: Sequence[Union[int, str]]
    ) -> TimeSeries:
        cols = list(ts.columns)
        selected_names: List[str] = []
        for item in selector:
            if isinstance(item, (int, np.integer)):
                selected_names.append(cols[int(item)])
            else:
                selected_names.append(str(item))
        return ts[selected_names]

    def _apply_darts_operations(
        self, ts: TimeSeries
    ) -> Tuple[TimeSeries, Optional[TimeSeries], Optional[TimeSeries]]:
        """
        Split ``ts`` into target, future dynamic covariates, and past dynamic
        covariates using :class:`darts.TimeSeries` methods and apply configured
        Darts transformations.
        """
        all_cols = list(ts.columns)

        fut_cov_ts: Optional[TimeSeries] = None
        past_cov_ts: Optional[TimeSeries] = None

        if self.feat_dynamic_real_components is not None:
            fut_cov_ts = self._select_components(ts, self.feat_dynamic_real_components)

        if self.past_feat_dynamic_real_components is not None:
            past_cov_ts = self._select_components(
                ts, self.past_feat_dynamic_real_components
            )

        if self.target_components is not None:
            target_ts = self._select_components(ts, self.target_components)
        else:
            excluded = set()
            if fut_cov_ts is not None:
                excluded.update(list(fut_cov_ts.columns))
            if past_cov_ts is not None:
                excluded.update(list(past_cov_ts.columns))
            remaining = [c for c in all_cols if c not in excluded]
            target_ts = ts[remaining] if len(remaining) > 0 else ts

        # 1. Optional target differencing via Darts TimeSeries.diff
        if self.diff_n is not None and int(self.diff_n) > 0:
            target_ts = target_ts.diff(n=int(self.diff_n), dropna=False)

        # 2. Optional Darts window_transform to generate past dynamic covariates
        if self.window_transforms is not None:
            wt_ts = target_ts.window_transform(
                transforms=self.window_transforms,
                treat_na=None,
                keep_non_transformed=False,
            )
            past_cov_ts = wt_ts if past_cov_ts is None else past_cov_ts.stack(wt_ts)

        # 3. Optional datetime attributes via Darts TimeSeries.add_datetime_attribute
        if self.datetime_attributes is not None:
            if not target_ts.has_datetime_index:
                self.raise_error(
                    "'datetime_attributes' requires a DatetimeIndex on the Darts TimeSeries "
                    "(specify 'freq' in dataset_args)."
                )
            attrs = (
                [self.datetime_attributes]
                if isinstance(self.datetime_attributes, str)
                else list(self.datetime_attributes)
            )
            for attr in attrs:
                with_dt = target_ts.add_datetime_attribute(str(attr))
                new_cols = [c for c in with_dt.columns if c not in target_ts.columns]
                dt_cov = with_dt[new_cols]
                fut_cov_ts = (
                    dt_cov if fut_cov_ts is None else fut_cov_ts.stack(dt_cov)
                )

        # 4. Optional country holidays via Darts TimeSeries.add_holidays
        if self.country_holidays is not None:
            if not target_ts.has_datetime_index:
                self.raise_error(
                    "'country_holidays' requires a DatetimeIndex on the Darts TimeSeries "
                    "(specify 'freq' in dataset_args)."
                )
            with_hol = target_ts.add_holidays(str(self.country_holidays))
            new_cols = [c for c in with_hol.columns if c not in target_ts.columns]
            hol_cov = with_hol[new_cols]
            fut_cov_ts = (
                hol_cov if fut_cov_ts is None else fut_cov_ts.stack(hol_cov)
            )

        # 5. Optional relative time index covariate
        if self.add_relative_index:
            n_steps = len(target_ts)
            rel_vals = np.linspace(0.0, 1.0, num=n_steps, dtype=np.float32)[:, np.newaxis]
            if target_ts.has_datetime_index:
                rel_ts = TimeSeries.from_times_and_values(
                    times=target_ts.time_index,
                    values=rel_vals,
                    columns=["relative_index"],
                    fill_missing_dates=False,
                )
            else:
                rel_ts = TimeSeries.from_values(
                    values=rel_vals,
                    columns=["relative_index"],
                )
            fut_cov_ts = rel_ts if fut_cov_ts is None else fut_cov_ts.stack(rel_ts)

        return target_ts, fut_cov_ts, past_cov_ts

    def _impute_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Impute NaN values column-wise using :class:`CausalMeanImputation` when
        ``self.impute_missing`` is enabled; otherwise replace NaNs with ``fillna_value``.
        """
        out = np.array(arr, dtype=np.float32, copy=True)
        if not np.isnan(out).any():
            return out
        if not self.impute_missing:
            out[np.isnan(out)] = float(self.fillna_value or 0.0)
            return out

        if out.ndim == 1:
            return self._imputer(out)
        for col_idx in range(out.shape[1]):
            col = out[:, col_idx]
            if np.isnan(col).any():
                out[:, col_idx] = self._imputer(col.copy())
        return out

    @staticmethod
    def _split_darts_ts(
        ts: TimeSeries, split_idx: int
    ) -> Tuple[Optional[TimeSeries], Optional[TimeSeries]]:
        """
        Split a :class:`darts.TimeSeries` into ``(past_ts, future_ts)`` at ``split_idx``
        using :meth:`TimeSeries.split_before` when ``0 < split_idx < len(ts)``.
        """
        n_steps = len(ts)
        if split_idx <= 0:
            return None, ts
        if split_idx >= n_steps:
            return ts, None
        split_point = ts.time_index[split_idx]
        return ts.split_before(split_point)

    def _extract_past_and_future(
        self, ts: TimeSeries
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Slice/pad ``ts`` into ``(past_vals, past_obs, past_is_pad, fut_vals, fut_obs, fut_is_pad)``
        using :meth:`TimeSeries.split_before`.
        """
        n_steps = len(ts)
        width = ts.width
        ctx_len = self.context_length
        pred_len = self.prediction_length

        if pred_len > 0:
            split_idx = max(0, min(n_steps - pred_len, ctx_len)) if n_steps > pred_len else n_steps
            past_ts, fut_ts = self._split_darts_ts(ts, split_idx)
        else:
            past_ts, fut_ts = ts, None

        past_raw = (
            past_ts.values(copy=True).astype(np.float32)
            if past_ts is not None
            else np.empty((0, width), dtype=np.float32)
        )
        fut_raw = (
            fut_ts.values(copy=True).astype(np.float32)
            if fut_ts is not None
            else np.empty((0, width), dtype=np.float32)
        )

        # Prepare past window of length ctx_len (left-pad if shorter, take tail if longer)
        past_is_pad = np.zeros((ctx_len,), dtype=bool)
        if len(past_raw) >= ctx_len:
            past_window = past_raw[-ctx_len:, :] if ctx_len > 0 else np.empty((0, width), dtype=np.float32)
            past_obs = ~np.isnan(past_window)
            past_vals = self._impute_array(past_window)
        else:
            pad_len = ctx_len - len(past_raw)
            past_is_pad[:pad_len] = True
            obs_tail = ~np.isnan(past_raw)
            imputed_tail = self._impute_array(past_raw) if len(past_raw) > 0 else past_raw
            fill_row = (
                imputed_tail[0:1, :]
                if len(imputed_tail) > 0
                else np.full((1, width), float(self.fillna_value or 0.0), dtype=np.float32)
            )
            pad_vals = np.repeat(fill_row, pad_len, axis=0)
            pad_obs = np.zeros((pad_len, width), dtype=bool)
            past_vals = np.concatenate([pad_vals, imputed_tail], axis=0)
            past_obs = np.concatenate([pad_obs, obs_tail], axis=0)

        # Prepare future window of length pred_len (right-pad if shorter, take head if longer)
        fut_is_pad = np.zeros((pred_len,), dtype=bool)
        if pred_len == 0:
            fut_vals = np.empty((0, width), dtype=np.float32)
            fut_obs = np.empty((0, width), dtype=bool)
        elif len(fut_raw) >= pred_len:
            fut_window = fut_raw[:pred_len, :]
            fut_obs = ~np.isnan(fut_window)
            fut_vals = np.where(fut_obs, fut_window, float(self.fillna_value or 0.0)).astype(
                np.float32
            )
        else:
            pad_len = pred_len - len(fut_raw)
            fut_is_pad[len(fut_raw) :] = True
            obs_head = ~np.isnan(fut_raw)
            clean_head = np.where(
                obs_head, fut_raw, float(self.fillna_value or 0.0)
            ).astype(np.float32)
            pad_vals = np.full(
                (pad_len, width), float(self.fillna_value or 0.0), dtype=np.float32
            )
            pad_obs = np.zeros((pad_len, width), dtype=bool)
            fut_vals = np.concatenate([clean_head, pad_vals], axis=0)
            fut_obs = np.concatenate([obs_head, pad_obs], axis=0)

        return past_vals, past_obs, past_is_pad, fut_vals, fut_obs, fut_is_pad

    def _build_sample_dict(self, index: int) -> Dict[str, torch.Tensor]:
        target_ts = self._target_ts_list[index]
        fut_cov_ts = self._future_cov_ts_list[index]
        past_cov_ts = self._past_cov_ts_list[index]

        (
            past_vals,
            past_obs,
            past_is_pad,
            fut_vals,
            fut_obs,
            fut_is_pad,
        ) = self._extract_past_and_future(target_ts)

        sample: Dict[str, torch.Tensor] = {
            "past_target": torch.tensor(past_vals, dtype=self.dtype),
            "past_observed_target": torch.tensor(past_obs, dtype=torch.bool),
            "past_is_pad": torch.tensor(past_is_pad, dtype=torch.bool),
        }

        if self.prediction_length > 0:
            sample["future_target"] = torch.tensor(fut_vals, dtype=self.dtype)
            sample["future_observed_target"] = torch.tensor(fut_obs, dtype=torch.bool)
            sample["future_is_pad"] = torch.tensor(fut_is_pad, dtype=torch.bool)

        if fut_cov_ts is not None:
            (
                cov_past_vals,
                cov_past_obs,
                _,
                cov_fut_vals,
                cov_fut_obs,
                _,
            ) = self._extract_past_and_future(fut_cov_ts)
            full_cov_vals = (
                np.concatenate([cov_past_vals, cov_fut_vals], axis=0)
                if self.prediction_length > 0
                else cov_past_vals
            )
            full_cov_obs = (
                np.concatenate([cov_past_obs, cov_fut_obs], axis=0)
                if self.prediction_length > 0
                else cov_past_obs
            )
            sample["feat_dynamic_real"] = torch.tensor(full_cov_vals, dtype=self.dtype)
            sample["observed_feat_dynamic_real"] = torch.tensor(
                full_cov_obs, dtype=torch.bool
            )

        if past_cov_ts is not None:
            p_cov_vals, p_cov_obs, _, _, _, _ = self._extract_past_and_future(
                past_cov_ts
            )
            sample["past_feat_dynamic_real"] = torch.tensor(
                p_cov_vals, dtype=self.dtype
            )
            sample["past_observed_feat_dynamic_real"] = torch.tensor(
                p_cov_obs, dtype=torch.bool
            )

        if target_ts.has_static_covariates:
            sc_vals = target_ts.static_covariates_values(copy=True)
            sample["static_covariates"] = torch.tensor(
                sc_vals, dtype=self.dtype
            ).flatten()

        return sample

    # ------------------------------------------------------------------
    # Darts TimeSeries Accessors
    # ------------------------------------------------------------------

    def get_timeseries(self, index: int) -> TimeSeries:
        """
        Return the underlying :class:`darts.TimeSeries` for sample ``index``.
        """
        return self.timeseries_list[index]

    def get_target_timeseries(self, index: int) -> TimeSeries:
        """
        Return the processed target :class:`darts.TimeSeries` for sample ``index``.
        """
        return self._target_ts_list[index]

    def get_all_timeseries(self) -> List[TimeSeries]:
        """
        Return all underlying :class:`darts.TimeSeries` instances in this dataset.
        """
        return list(self.timeseries_list)

    def get_feature_dims(self) -> Dict[str, int]:
        """
        Return a dictionary with ``target_dim``, ``feat_dynamic_real_dim``, and
        ``past_feat_dynamic_real_dim``.
        """
        return {
            "target_dim": self.target_dim,
            "feat_dynamic_real_dim": self.feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": self.past_feat_dynamic_real_dim,
        }

    # ------------------------------------------------------------------
    # PyTorch Dataset & BaseDataset Interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.samples[index], self.labels[index]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all target time-series datapoints and labels as NumPy arrays of shape
        ``(n_datapoints, sequence_length, target_dim)`` and ``(n_datapoints, n_labels)``.
        """
        if len(self.samples) == 0:
            return np.empty((0, 0, 0)), np.empty((0, 0))

        series_arrays = []
        for s in self.samples:
            past = s["past_target"].detach().cpu().numpy()
            if "future_target" in s:
                fut = s["future_target"].detach().cpu().numpy()
                series_arrays.append(np.concatenate([past, fut], axis=0))
            else:
                series_arrays.append(past)

        labels_array = np.array([l.detach().cpu().numpy() for l in self.labels])
        return np.stack(series_arrays, axis=0), labels_array

    def get_norm_data(self) -> np.ndarray:
        """
        Return 2-D array of shape ``(n_datapoints * sequence_length, target_dim)``
        for fitting feature-wise normalizers (scalers).
        """
        if len(self.samples) == 0:
            return np.empty((0, 0))
        data_3d, _ = self.get_data()
        return data_3d.reshape(-1, data_3d.shape[-1])

    def get_datapoint_shape(self) -> tuple:
        """
        Return the shape of a single target series ``(sequence_length, target_dim)``.
        """
        if len(self.samples) == 0:
            return ()
        return (self.total_window_length, self.target_dim)
