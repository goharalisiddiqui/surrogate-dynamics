from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.neighbors import KernelDensity

from collective_encoder.utils import compute_mfpt_matrix
from surrogate_dynamics.testplotters.simple import TimeseriesSimplePlotter
from surrogate_dynamics.utils import periodic_average, periodic_norm_distance

ALA2_MINIMA_POSITIONS = np.array([
    [-1.3144, -0.3401],
    [-1.2843, 2.6713],
    [-2.6904, 2.7835],
    [-1.4946, -3.0109],
    [-2.6308, -2.9918],
    [3.0341, 2.7583],
    [1.2012, 0.0580],
    [3.0745, -3.0108],
    [3.0320, -0.6649],
    [1.0283, -2.7108],
    [1.2054, 2.9642],
    [3.0375, -1.9775],
])


class MSMPlotter(TimeseriesSimplePlotter):
    """
    Test plotter that estimates a 2-D Free Energy Surface (FES) via Gaussian Kernel
    Density Estimation (KDE), identifies metastable local minima via multi-start
    BFGS optimization, optionally matches discovered minima against reference
    positions via the Hungarian algorithm, and constructs a Markov State Model (MSM)
    with transition counts, transition probability matrix, and Mean First Passage
    Time (MFPT) matrix.

    Inherits from :class:`~surrogate_dynamics.testplotters.simple.TimeseriesSimplePlotter`
    to support decoding latent trajectory predictions via ``embedding_model`` as well
    as cosine/sine angle resolution.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary supporting all :class:`TimeseriesSimplePlotter`
        keys plus:
        - ``x`` (str, optional): Name of the first 2-D coordinate in selections.
        - ``y`` (str, optional): Name of the second 2-D coordinate in selections.
        - ``msm_plots`` (list of dict, optional): List of plot specifications, each
          with keys ``{"name": str, "x": str, "y": str}``, allowing FES and MSM
          evaluation on multiple coordinate pairs (e.g., predicted and ground-truth).
        - ``periodic`` (bool, default True): Whether coordinates are periodic in
          ``[-pi, pi]``.
        - ``kbt`` (float, default 0.415): Thermal energy factor $k_B T$ in kJ/mol.
        - ``kde_sigma`` (float, default 0.2): Gaussian KDE bandwidth.
        - ``kde_bin_x`` (int, default 200): Number of grid points along X for FES.
        - ``kde_bin_y`` (int, default 200): Number of grid points along Y for FES.
        - ``n_BFGS_runs`` (int, default 100): Number of random-restart BFGS runs
          used to locate local minima on the FES.
        - ``minima_tolerance`` (float, default 0.1): Distance tolerance (in radians
          when periodic) for deduplicating and matching minima.
        - ``lag`` (int, default 1): Lag time (in trajectory steps) for MSM transition
          matrix and MFPT estimation.
        - ``reference_minima`` (str or list or numpy.ndarray or None, default "ALA2"):
          Reference minima positions for recall evaluation (``"ALA2"`` uses the
          12 reference Alanine Dipeptide minima, a custom ``(M, 2)`` array/list, or
          ``None`` to skip reference matching).
        - ``n_minima_compare`` (int, default 3): Number of top reference minima to
          compare against when ``reference_minima`` is provided.
    kwargs : dict
        Additional keyword arguments forwarded to :class:`TimeseriesSimplePlotter`.
    """

    _IDENTIFIER = "MSM"
    _OPTIONAL_ARGS = TimeseriesSimplePlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "x": None,
        "y": None,
        "msm_plots": None,
        "periodic": True,
        "kbt": 0.415,
        "kde_sigma": 0.2,
        "kde_bin_x": 200,
        "kde_bin_y": 200,
        "n_BFGS_runs": 100,
        "minima_tolerance": 0.1,
        "lag": 1,
        "reference_minima": "ALA2",
        "n_minima_compare": 3,
    })

    def __init__(self, args: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize ``MSMPlotter`` and resolve reference minima coordinates.

        Parameters
        ----------
        args : dict, optional
            Plotter configuration dictionary.
        **kwargs
            Runtime keyword arguments forwarded to :class:`TimeseriesSimplePlotter`.
        """
        super().__init__(args=args, **kwargs)
        self.ref_minima_array = self._resolve_reference_minima(self.reference_minima)
        if (
            self.ref_minima_array is not None
            and self.n_minima_compare > len(self.ref_minima_array)
        ):
            self.log_warn(
                f"n_minima_compare ({self.n_minima_compare}) is greater than the "
                f"number of reference minima ({len(self.ref_minima_array)}). "
                f"Setting n_minima_compare to {len(self.ref_minima_array)}."
            )
            self.n_minima_compare = len(self.ref_minima_array)

    def _resolve_reference_minima(
        self, ref: Union[str, List[List[float]], np.ndarray, None]
    ) -> Optional[np.ndarray]:
        """
        Resolve ``reference_minima`` into a 2-D NumPy array of shape ``(M, 2)`` or ``None``.

        Parameters
        ----------
        ref : str or list or numpy.ndarray or None
            Reference minima specification.

        Returns
        -------
        numpy.ndarray or None
            Array of shape ``(M, 2)`` or ``None``.
        """
        if ref is None:
            return None
        if isinstance(ref, str):
            if ref.upper() == "ALA2":
                return ALA2_MINIMA_POSITIONS.copy()
            self.raise_error(
                f"Unknown reference_minima preset '{ref}'. Supported preset is 'ALA2', "
                "a 2-D list/array of coordinates, or None."
            )
        arr = np.asarray(ref, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            self.raise_error(
                f"reference_minima must have shape (M, 2), got {arr.shape}."
            )
        return arr

    def _build_kde(self, x: np.ndarray, y: np.ndarray) -> KernelDensity:
        """
        Fit a 2-D Gaussian Kernel Density Estimator on ``(x, y)`` samples.

        Parameters
        ----------
        x : numpy.ndarray
            1-D array of first coordinate values.
        y : numpy.ndarray
            1-D array of second coordinate values.

        Returns
        -------
        sklearn.neighbors.KernelDensity
            Fitted 2-D Gaussian KDE instance.
        """
        coords = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
        return KernelDensity(kernel="gaussian", bandwidth=float(self.kde_sigma)).fit(
            coords
        )

    def _plot_kde(
        self,
        kde: KernelDensity,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        xlabel: str = r"$\Phi$",
        ylabel: str = r"$\Psi$",
    ) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
        """
        Evaluate the 2-D Free Energy Surface on a grid and generate a contour plot.

        Parameters
        ----------
        kde : sklearn.neighbors.KernelDensity
            Fitted 2-D KDE model.
        x_vals : numpy.ndarray
            Observed X coordinates (used to determine grid bounds when non-periodic).
        y_vals : numpy.ndarray
            Observed Y coordinates (used to determine grid bounds when non-periodic).
        xlabel : str, optional
            X-axis label.
        ylabel : str, optional
            Y-axis label.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes, numpy.ndarray)
            Generated figure, axes, and 2-D FES grid array.
        """
        if self.periodic:
            x_min, x_max = -np.pi, np.pi
            y_min, y_max = -np.pi, np.pi
        else:
            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

        xg = np.linspace(x_min, x_max, int(self.kde_bin_x))
        yg = np.linspace(y_min, y_max, int(self.kde_bin_y))
        x_grid, y_grid = np.meshgrid(xg, yg)

        grid_pts = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        fes = float(self.kbt) * -kde.score_samples(grid_pts).reshape(x_grid.shape)
        fes -= fes.min()

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        cf = ax.contourf(x_grid, y_grid, fes, levels=30, cmap=cm.viridis)
        fig.colorbar(cf, ax=ax, label="Free Energy (kJ/mol)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        if self.periodic:
            ticks = [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi]
            tick_labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
        plt.tight_layout()
        return fig, ax, fes

    def _find_kde_minima(
        self,
        kde: KernelDensity,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify local minima of the negative log-density via multi-start BFGS.

        Parameters
        ----------
        kde : sklearn.neighbors.KernelDensity
            Fitted 2-D KDE model.
        bounds : tuple of ((float, float), (float, float))
            Sampling range ``((x_min, x_max), (y_min, y_max))`` for initial guesses.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            - ``min_points``: Deduplicated minima coordinates of shape ``(K, 2)``,
              sorted from lowest to highest objective value.
            - ``min_vals``: Objective values ``-log P(x)`` at each minimum.
        """
        def objective(pt: np.ndarray) -> float:
            if self.periodic:
                pt = ((pt + np.pi) % (2.0 * np.pi)) - np.pi
            return float(-kde.score_samples(pt.reshape(1, -1))[0])

        (x_min, x_max), (y_min, y_max) = bounds
        raw_points: List[np.ndarray] = []
        for _ in range(int(self.n_BFGS_runs)):
            x0 = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
            ])
            res = minimize(objective, x0, method="BFGS")
            if res.success:
                pt = res.x
                if self.periodic:
                    pt = ((pt + np.pi) % (2.0 * np.pi)) - np.pi
                raw_points.append(pt)

        if not raw_points:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

        clusters: List[List[np.ndarray]] = []
        tol = float(self.minima_tolerance)
        for p in raw_points:
            matched_cluster = False
            for cluster in clusters:
                if any(
                    (
                        periodic_norm_distance(np.asarray(p), np.asarray(r))
                        if self.periodic
                        else float(np.linalg.norm(np.asarray(p) - np.asarray(r)))
                    )
                    <= tol
                    for r in cluster
                ):
                    cluster.append(p)
                    matched_cluster = True
                    break
            if not matched_cluster:
                clusters.append([p])

        if self.periodic:
            min_points = np.array([periodic_average(c) for c in clusters])
        else:
            min_points = np.array([np.mean(c, axis=0) for c in clusters])

        min_vals = np.array([objective(p) for p in min_points])
        order = np.argsort(min_vals)
        return min_points[order], min_vals[order]

    def _match_reference_minima(
        self, minima_model: np.ndarray, minima_ref: np.ndarray
    ) -> Tuple[np.ndarray, List[float], np.ndarray]:
        """
        Match discovered model minima to reference minima via the Hungarian algorithm.

        Parameters
        ----------
        minima_model : numpy.ndarray
            Model-discovered minima of shape ``(N, 2)``.
        minima_ref : numpy.ndarray
            Reference minima of shape ``(M, 2)``.

        Returns
        -------
        tuple of (numpy.ndarray, list of float, numpy.ndarray)
            - ``matched_minima``: Model minima within ``self.minima_tolerance`` of
              their assigned reference minimum.
            - ``correct_dists``: Distances for matched pairs within tolerance.
            - ``all_dists``: Assigned pairwise distances for all Hungarian matches.
        """
        n_model = len(minima_model)
        n_ref = len(minima_ref)
        if n_model == 0 or n_ref == 0:
            return np.empty((0, 2)), [], np.empty((0,))

        cost_matrix = np.zeros((n_model, n_ref), dtype=np.float64)
        for i in range(n_model):
            for j in range(n_ref):
                if self.periodic:
                    cost_matrix[i, j] = periodic_norm_distance(
                        minima_model[i], minima_ref[j], period=2.0 * np.pi
                    )
                else:
                    cost_matrix[i, j] = float(
                        np.linalg.norm(minima_model[i] - minima_ref[j])
                    )

        model_indices, ref_indices = linear_sum_assignment(cost_matrix)
        matched_minima: List[np.ndarray] = []
        correct_dists: List[float] = []
        dists: List[float] = []
        tol = float(self.minima_tolerance)
        for m_idx, r_idx in zip(model_indices, ref_indices):
            dist = float(cost_matrix[m_idx, r_idx])
            dists.append(dist)
            if dist <= tol:
                matched_minima.append(minima_model[m_idx])
                correct_dists.append(dist)

        matched_arr = (
            np.array(matched_minima, dtype=np.float64)
            if matched_minima
            else np.empty((0, 2), dtype=np.float64)
        )
        return matched_arr, correct_dists, np.array(dists, dtype=np.float64)

    def _resolve_msm_plot_specs(self) -> List[Dict[str, str]]:
        """
        Determine the list of 2-D coordinate pairs ``[{"name": ..., "x": ..., "y": ...}]``
        on which to run FES and MSM analysis.

        Returns
        -------
        list of dict
            Plot specification dictionaries.
        """
        if self.msm_plots:
            return list(self.msm_plots)
        if self.x is not None and self.y is not None:
            return [{"name": f"{self.x}_{self.y}", "x": self.x, "y": self.y}]
        if self.plots_2dscatter_cb:
            return [
                {
                    "name": spec.get("name", f"{spec['x']}_{spec['y']}"),
                    "x": spec["x"],
                    "y": spec["y"],
                }
                for spec in self.plots_2dscatter_cb
                if "x" in spec and "y" in spec
            ]
        keys = list(getattr(self, "vals", {}).keys())
        if len(keys) >= 2:
            return [{"name": f"{keys[0]}_{keys[1]}", "x": keys[0], "y": keys[1]}]
        return []

    def _run_msm_for_pair(
        self, x_vals: np.ndarray, y_vals: np.ndarray, suffix: str = ""
    ) -> None:
        """
        Execute KDE FES estimation, BFGS minima finding, reference matching, and
        MFPT matrix plotting for a single 2-D coordinate pair.

        Parameters
        ----------
        x_vals : numpy.ndarray
            1-D array of first coordinate values.
        y_vals : numpy.ndarray
            1-D array of second coordinate values.
        suffix : str, optional
            Filename/metric suffix when multiple coordinate pairs are evaluated.
        """
        tag_suffix = f"_{suffix}" if suffix else ""
        self._save_data(x_vals, f"x{tag_suffix}")
        self._save_data(y_vals, f"y{tag_suffix}")

        kde = self._build_kde(x_vals, y_vals)
        fig, ax, fes = self._plot_kde(kde, x_vals, y_vals)
        self._save_data(fes, f"fes{tag_suffix}")

        bounds = (
            ((-np.pi, np.pi), (-np.pi, np.pi))
            if self.periodic
            else (
                (float(np.min(x_vals)), float(np.max(x_vals))),
                (float(np.min(y_vals)), float(np.max(y_vals))),
            )
        )
        min_points, _ = self._find_kde_minima(kde, bounds)
        self.log_info(f"Number of local minima found{tag_suffix}: {len(min_points)}")
        self.set_metric(f"n_minima_found{tag_suffix}", len(min_points))
        self._save_data(min_points, f"minima_positions{tag_suffix}")

        if len(min_points) > 0:
            ax.scatter(
                min_points[:, 0],
                min_points[:, 1],
                facecolors="w",
                edgecolors="r",
                s=200,
                label="Local Minima",
            )
            for idx, (mx, my) in enumerate(min_points):
                ax.text(
                    mx,
                    my,
                    str(idx + 1),
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                )
            ax.legend(bbox_to_anchor=(1, 1.1))

        self.log_image(fig, f"kde_estimation{tag_suffix}")
        plt.close(fig)

        msm_centers = min_points
        if self.ref_minima_array is not None and len(min_points) > 0:
            n_comp = min(int(self.n_minima_compare), len(min_points))
            correct_minima, _, dists = self._match_reference_minima(
                minima_model=min_points,
                minima_ref=self.ref_minima_array[:n_comp],
            )
            n_correct = len(correct_minima)
            recall = float(n_correct) / float(n_comp) if n_comp > 0 else 0.0
            mean_dist = float(np.mean(dists)) if len(dists) > 0 else float("nan")

            self.set_metric(f"n_minima_matched{tag_suffix}", n_correct)
            self.set_metric(f"minima_recall{tag_suffix}", recall)
            self.set_metric(f"minima_mean_dist{tag_suffix}", mean_dist)
            self.log_info(
                f"MSM minima matching{tag_suffix}: {n_correct}/{n_comp} within "
                f"{self.minima_tolerance} (mean dist={mean_dist:.4f})"
            )
            if len(correct_minima) > 1:
                msm_centers = correct_minima

        if len(msm_centers) > 1:
            traj_2d = np.stack([x_vals.reshape(-1), y_vals.reshape(-1)], axis=-1)
            transition_counts, transition_matrix, mfpt_matrix = compute_mfpt_matrix(
                traj_2d, msm_centers, lag=int(self.lag)
            )
            self._save_data(transition_counts, f"transition_counts{tag_suffix}")
            self._save_data(transition_matrix, f"transition_matrix{tag_suffix}")
            self._save_data(mfpt_matrix, f"mfpt_matrix{tag_suffix}")

            for mat_name, mat_data, label in [
                ("transition_counts", transition_counts, "Transition Counts"),
                ("transition_matrix", transition_matrix, "Transition Matrix"),
                ("mfpt_matrix", mfpt_matrix, "MFPT Matrix"),
            ]:
                fig_m, _ = self._plot_matrix(
                    mat_data, label=label, tag="Source_Target"
                )
                self.log_image(fig_m, f"{mat_name}{tag_suffix}")
                plt.close(fig_m)
        else:
            self.log_info(
                f"Fewer than 2 minima available for MSM{tag_suffix}; skipping MFPT computation."
            )

    def plot(
        self,
        data: Any,
        latent: Any,
        pred: Any,
        labels: Any,
        meta: Any,
    ) -> None:
        """
        Decode predictions, run standard selections/scatter plots via
        :class:`TimeseriesSimplePlotter`, and perform 2-D KDE FES + MSM analysis.

        Parameters
        ----------
        data : Any
            Accumulated inputs.
        latent : dict or numpy.ndarray
            Accumulated latent representations.
        pred : numpy.ndarray
            Accumulated latent predictions.
        labels : dict or numpy.ndarray
            Accumulated ground-truth labels.
        meta : dict
            Accumulated metadata.
        """
        super().plot(data, latent, pred, labels, meta)
        vals = getattr(self, "vals", {})
        if not vals:
            return

        specs = self._resolve_msm_plot_specs()
        for spec in specs:
            x_key, y_key = spec["x"], spec["y"]
            if x_key not in vals or y_key not in vals:
                self.log_warn(
                    f"MSMPlotter: keys '{x_key}' or '{y_key}' not found in selected "
                    f"values {list(vals.keys())}. Skipping."
                )
                continue
            suffix = spec.get("name", "") if len(specs) > 1 else ""
            self._run_msm_for_pair(
                np.asarray(vals[x_key], dtype=np.float64).reshape(-1),
                np.asarray(vals[y_key], dtype=np.float64).reshape(-1),
                suffix=suffix,
            )
