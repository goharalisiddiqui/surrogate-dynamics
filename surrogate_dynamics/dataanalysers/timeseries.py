from typing import Any, Dict, List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from collective_encoder.dataanalysers.simple import SimpleDataAnalyser


class TimeseriesDataAnalyser(SimpleDataAnalyser):
    """
    Data analyser for visualizing 3-D time-series datasets of shape
    ``(n_datapoints, sequence_length, n_features)`` (as well as 2-D ``COLVAR`` datasets).

    Inherits from :class:`~collective_encoder.dataanalysers.simple.SimpleDataAnalyser`
    (which subclasses :class:`~collective_encoder.dataanalysers.base.BaseDataAnalyser`
    and :class:`~collective_encoder.common.plotter.BasePlotter`).

    In addition to standard :class:`BasePlotter` 2-D scatter plots
    (``plots_2dscatter_cb``), correlation matrices (``correlations``), and
    ``transformed_values``, this analyser generates time-series visualizations
    via :meth:`_log_image`:
    1. **Sample Trajectory Line Plots** (``trajectories_<label>.png``):
       Plots ``n_samples`` time-series sequences of length ``sequence_length`` for
       each selected feature, marking ``input_chunk_length`` and
       ``output_chunk_length`` boundaries when present in ``datamodule_args``.
    2. **Flattened Time-Series Overview** (``timeseries_<label>.png``):
       Plots concatenated values across datapoints with alternating colors per
       sequence of length ``sequence_length``.
    3. **Feature / Label Distribution Histograms** (``histograms_<label>.png``):
       Plots distribution histograms annotated with mean, std, min, and max.
    """

    _IDENTIFIER = "TIMESERIES"
    _COMPATIBLE_DATASET_TYPES = ["TIMESERIES", "COLVAR"]
    _OPTIONAL_ARGS = SimpleDataAnalyser._OPTIONAL_ARGS | {
        "n_samples": 5,
        "max_features": 10,
        "plot_trajectories": True,
        "plot_timeseries": True,
        "plot_histograms": True,
    }

    plot_correlation = SimpleDataAnalyser._plot_correlation

    def after_correlation_matrix(
        self,
        corr: Dict[str, Any],
        corr_matrix: np.ndarray,
        fig: plt.Figure,
        axes: np.ndarray,
    ) -> None:
        prev = corr.get("name", f"correlation_{corr['x']}_{corr['y']}")
        if getattr(self, "label", ""):
            corr["name"] = f"{prev}_{self.label}"

    def parse_data(
        self, data: Any
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Extract and reshape ``(datapoints, labelpoints)`` from a ``TIMESERIES``
        (or ``COLVAR``) dataset so that:
        - ``self._traj_array`` retains the 3-D ``(n_datapoints, sequence_length, n_features)``
          structure for sample trajectory plotting.
        - Returned ``datapoints`` is a 2-D array of shape
          ``(n_datapoints * sequence_length, n_features)`` compatible with
          :meth:`BasePlotter._parse_selection`.
        - Returned ``labelpoints`` is a 2-D array repeated across ``sequence_length``
          when labels are per-sequence ``(n_datapoints, n_labels)``, so selected
          labels and features share the same 1-D length in ``self.vals``.
        """
        datapoints, labelpoints = data.get_data()
        arr = np.asarray(datapoints)
        if arr.ndim == 2:
            seq_len = self._resolve_sequence_length(default=1)
            if seq_len > 1 and len(arr) >= seq_len and len(arr) % seq_len == 0:
                self._traj_array = arr.reshape(-1, seq_len, arr.shape[-1])
            else:
                self._traj_array = arr[:, np.newaxis, :]
            datapoints_2d = arr
        elif arr.ndim == 3:
            self._traj_array = arr
            n_dp, seq_len, n_feat = arr.shape
            datapoints_2d = arr.reshape(n_dp * seq_len, n_feat)
        else:
            self.raise_error(
                f"Expected 2-D or 3-D datapoints array from dataset.get_data(), got shape {arr.shape}."
            )

        lbl_arr = np.asarray(labelpoints)
        if lbl_arr.ndim == 1:
            lbl_arr = lbl_arr[:, np.newaxis]
        elif lbl_arr.ndim == 3:
            lbl_arr = lbl_arr.reshape(-1, lbl_arr.shape[-1])

        if (
            self._traj_array.ndim == 3
            and self._traj_array.shape[1] > 1
            and len(lbl_arr) == self._traj_array.shape[0]
            and len(datapoints_2d) == self._traj_array.shape[0] * self._traj_array.shape[1]
        ):
            lbl_arr = np.repeat(lbl_arr, self._traj_array.shape[1], axis=0)

        return datapoints_2d, lbl_arr

    def _resolve_sequence_length(self, default: int = 1) -> int:
        """
        Determine sequence length from ``self._traj_array`` or ``datamodule_args``.
        """
        if hasattr(self, "_traj_array") and self._traj_array is not None and self._traj_array.ndim == 3:
            if self._traj_array.shape[1] > 1:
                return int(self._traj_array.shape[1])

        dm_args = self.datamodule_args or {}
        if dm_args.get("sequence_length") is not None:
            return int(dm_args["sequence_length"])
        in_chunk = dm_args.get("input_chunk_length")
        out_chunk = dm_args.get("output_chunk_length")
        if in_chunk is not None and out_chunk is not None:
            n_seq = dm_args.get("n_seq_per_sample", 1)
            return int(in_chunk) + int(out_chunk) * int(n_seq)
        return default

    def write_data(self, data: Any, label: str = "") -> None:
        """
        Parse selections (defaulting to the first ``max_features`` data channels
        when neither ``data_selection`` nor ``labels_selection`` is specified)
        and generate all configured plots.
        """
        if data is None or len(data) == 0:
            self.log_warn(f"No data available to analyse for split '{label}'.")
            return

        self.label = label
        datapoints, labelpoints = self.parse_data(data)

        orig_data_sel = self.data_selection
        if self.data_selection is None and self.labels_selection is None:
            n_features = datapoints.shape[1] if datapoints.ndim >= 2 else 1
            n_to_plot = (
                min(n_features, int(self.max_features))
                if self.max_features
                else n_features
            )
            self.data_selection = {f"feature_{i}": i for i in range(n_to_plot)}

        self._selected_data_map: Dict[str, int] = (
            {k: v for k, v in self.data_selection.items() if isinstance(v, int)}
            if isinstance(self.data_selection, dict)
            else {}
        )

        try:
            super().write_data(data, label=label)
        finally:
            self.data_selection = orig_data_sel

    def plot_vals(self) -> None:
        """
        Execute standard :class:`BasePlotter` 2-D scatter & correlation plots
        followed by time-series trajectory, flattened series, and histogram plots.
        """
        super().plot_vals()

        if not hasattr(self, "vals") or not self.vals:
            return

        suffix = f"_{self.label}" if getattr(self, "label", "") else ""

        if (
            self.plot_trajectories
            and hasattr(self, "_traj_array")
            and self._traj_array is not None
            and self._traj_array.shape[1] > 1
            and getattr(self, "_selected_data_map", None)
        ):
            self._plot_sample_trajectories(
                self._traj_array, self._selected_data_map, suffix=suffix
            )

        if self.plot_timeseries:
            seq_len = self._resolve_sequence_length(default=1)
            self._plot_flattened_timeseries(self.vals, seq_len=seq_len, suffix=suffix)

        if self.plot_histograms:
            self._plot_histograms(self.vals, suffix=suffix)

    def _plot_sample_trajectories(
        self,
        traj_array: np.ndarray,
        feature_map: Dict[str, int],
        suffix: str = "",
    ) -> None:
        """
        Plot ``n_samples`` time-series trajectories of length ``sequence_length``
        for each selected data feature.
        """
        n_total, seq_len, _ = traj_array.shape
        n_samples = min(int(self.n_samples), n_total)
        sample_indices = np.linspace(0, n_total - 1, num=n_samples, dtype=int)

        dm_args = self.datamodule_args or {}
        input_chunk = dm_args.get("input_chunk_length", None)
        output_chunk = dm_args.get("output_chunk_length", None)
        n_seq = dm_args.get("n_seq_per_sample", 1)

        n_feats = len(feature_map)
        fig, axes = plt.subplots(
            n_feats,
            1,
            figsize=(8, max(3, 2.5 * n_feats)),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes[:, 0]
        timesteps = np.arange(seq_len)

        for ax_idx, (feat_name, feat_idx) in enumerate(feature_map.items()):
            ax = axes_flat[ax_idx]
            for s_idx in sample_indices:
                ax.plot(
                    timesteps,
                    traj_array[s_idx, :, feat_idx],
                    alpha=0.8,
                    linewidth=1.2,
                    label=f"sample {s_idx}",
                )

            if input_chunk is not None and 0 < input_chunk < seq_len:
                ax.axvline(
                    input_chunk - 0.5,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    label="input/output split" if ax_idx == 0 else None,
                )
                ax.axvspan(0, input_chunk - 0.5, color="tab:blue", alpha=0.06)
                ax.axvspan(
                    input_chunk - 0.5, seq_len - 1, color="tab:orange", alpha=0.06
                )

                if output_chunk is not None and n_seq and n_seq > 1:
                    for step in range(1, n_seq):
                        boundary = input_chunk + step * output_chunk - 0.5
                        if boundary < seq_len - 1:
                            ax.axvline(
                                boundary,
                                color="gray",
                                linestyle=":",
                                linewidth=0.8,
                            )

            ax.set_ylabel(feat_name)
            ax.grid(True, linestyle="--", alpha=0.3)
            if ax_idx == 0:
                ax.legend(
                    loc="upper right", fontsize="small", ncol=min(n_samples + 1, 3)
                )

        axes_flat[-1].set_xlabel("Time step")
        title_label = getattr(self, "label", "")
        fig.suptitle(
            f"Sample Time-Series Trajectories ({title_label})"
            if title_label
            else "Sample Time-Series Trajectories",
            fontsize=11,
        )
        fig.tight_layout()
        self._log_image(fig, f"trajectories{suffix}")
        plt.close(fig)

    def _plot_flattened_timeseries(
        self,
        vals: Dict[str, np.ndarray],
        seq_len: int,
        suffix: str = "",
    ) -> None:
        """
        Plot concatenated 1-D values across datapoints with alternating colors per
        sequence of length ``seq_len``.
        """
        first_series = next(iter(vals.values()))
        total_points = len(first_series)

        if seq_len > 1:
            colors = ["red"] * seq_len + ["blue"] * seq_len
            colors = colors * (total_points // (2 * seq_len) + 1)
            colors = colors[:total_points]
        else:
            colors = "blue"

        n_vars = len(vals)
        fig, axes = plt.subplots(
            n_vars,
            1,
            figsize=(8, max(3, 2.2 * n_vars)),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes[:, 0]

        for i, (var_name, values) in enumerate(vals.items()):
            ax = axes_flat[i]
            c = colors[: len(values)] if isinstance(colors, list) else colors
            ax.scatter(range(len(values)), values, marker="+", s=5, c=c)
            ax.set_ylabel(var_name)
            ax.grid(True, linestyle="--", alpha=0.3)

        axes_flat[-1].set_xlabel("Flattened time index")
        fig.tight_layout()
        self._log_image(fig, f"timeseries{suffix}")
        plt.close(fig)

    def _plot_histograms(
        self,
        vals: Dict[str, np.ndarray],
        suffix: str = "",
    ) -> None:
        """
        Plot distribution histograms with mean/std/min/max summary annotations.
        """
        n_vars = len(vals)
        fig, axes = plt.subplots(
            n_vars,
            1,
            figsize=(6, max(3, 2.2 * n_vars)),
            squeeze=False,
        )
        axes_flat = axes[:, 0]

        for i, (var_name, values) in enumerate(vals.items()):
            ax = axes_flat[i]
            finite_vals = values[np.isfinite(values)]
            if len(finite_vals) == 0:
                continue
            mean_val = float(np.mean(finite_vals))
            std_val = float(np.std(finite_vals))
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))

            ax.hist(
                finite_vals,
                bins=50,
                color="steelblue",
                edgecolor="black",
                alpha=0.75,
                density=True,
            )
            ax.axvline(
                mean_val,
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                label=f"mean={mean_val:.3g}",
            )
            stats_label = (
                f"mean={mean_val:.3g}, std={std_val:.3g}\n"
                f"min={min_val:.3g}, max={max_val:.3g}"
            )
            ax.text(
                0.98,
                0.92,
                stats_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            ax.set_xlabel(var_name)
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.3)

        title_label = getattr(self, "label", "")
        fig.suptitle(
            f"Feature Distributions ({title_label})"
            if title_label
            else "Feature Distributions",
            fontsize=11,
        )
        fig.tight_layout()
        self._log_image(fig, f"histograms{suffix}")
        plt.close(fig)
