from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from collective_encoder.testplotters.base import BaseTestPlotter
from surrogate_dynamics.utils import calculate_transfer_matrix


class TransferOperatorPlotter(BaseTestPlotter):
    """
    Test plotter that estimates the linear transfer (Koopman) operator
    $K(\tau) = C_0^{-1} C_\tau$ on predicted latent trajectories, computes its
    eigenvalue spectrum and eigenvectors, and evaluates implied relaxation
    timescales $t_i = -\tau / \ln |\lambda_i|$.

    Inherits from :class:`~collective_encoder.testplotters.base.BaseTestPlotter`
    and operates directly in latent space without requiring a decoder.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary supporting all :class:`BaseTestPlotter` keys plus:
        - ``lag`` (int, default 1): Lag time $\tau$ in trajectory frames for primary
          transfer operator estimation.
        - ``lags`` (list of int, optional): Optional sequence of lag times used to
          plot implied timescale convergence vs. lag time.
        - ``n_modes`` (int, default 5): Maximum number of slowest relaxation modes
          to display and record as scalar metrics.
        - ``center_data`` (bool, default False): Whether to mean-center trajectory
          features before computing $C_0$ and $C_\tau$.
        - ``skip_stationary_mode`` (bool, default True): Whether to exclude the
          leading eigenvalue $\lambda_0$ when computing implied timescales (matches
          the legacy ``eigenvalues[1:]`` convention).
    kwargs : dict
        Additional keyword arguments forwarded to :class:`BaseTestPlotter`.
    """

    _IDENTIFIER = "TRANSFER_OPERATOR"
    _OPTIONAL_ARGS = BaseTestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "lag": 1,
        "lags": None,
        "n_modes": 5,
        "center_data": False,
        "skip_stationary_mode": True,
    })

    def collection_list(self) -> List[str]:
        """
        Specify data components collected across test batches.

        Returns
        -------
        list of str
            Collects ``["pred", "latent"]``.
        """
        return ["pred", "latent"]

    def _prepare_trajectory(self, arr: Any) -> Optional[np.ndarray]:
        """
        Convert a collected trajectory tensor or array into a 2-D ``(T, D)`` NumPy array.

        Parameters
        ----------
        arr : Any
            Accumulated trajectory array or dictionary.

        Returns
        -------
        numpy.ndarray or None
            2-D NumPy array of shape ``(n_frames, n_features)`` or ``None``.
        """
        if arr is None:
            return None
        if isinstance(arr, dict):
            if len(arr) == 0:
                return None
            arr = np.column_stack(
                [self._convert_to_numpy(v).reshape(-1) for v in arr.values()]
            )
        else:
            arr = self._convert_to_numpy(arr)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if self.center_data:
            arr = arr - np.mean(arr, axis=0, keepdims=True)
        return arr

    def _estimate_spectrum(
        self, trajectory: np.ndarray, lag: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the transfer matrix at ``lag``, compute sorted eigenvalues and
        eigenvectors, and calculate implied timescales.

        Parameters
        ----------
        trajectory : numpy.ndarray
            2-D trajectory array of shape ``(T, D)``.
        lag : int
            Lag time in frames.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            - ``eigenvalues``: Complex eigenvalues sorted descending by magnitude.
            - ``eigenvectors``: Corresponding column eigenvectors.
            - ``its``: Implied timescales in units of trajectory frames.
        """
        transfer_matrix = calculate_transfer_matrix(trajectory, lag=int(lag))
        eigenvalues, eigenvectors = np.linalg.eig(transfer_matrix)

        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        evals_for_its = (
            eigenvalues[1:]
            if (self.skip_stationary_mode and len(eigenvalues) > 1)
            else eigenvalues
        )
        mags = np.clip(np.abs(evals_for_its), 1e-12, 1.0 - 1e-12)
        its = -float(lag) / np.log(mags)
        return eigenvalues, eigenvectors, its

    def _plot_eigenvalue_spectrum(self, eigenvalues: np.ndarray) -> None:
        """
        Plot the transfer operator eigenvalues in the complex plane with the unit circle.

        Parameters
        ----------
        eigenvalues : numpy.ndarray
            1-D array of complex eigenvalues.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        theta = np.linspace(0.0, 2.0 * np.pi, 300)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            linestyle="--",
            color="gray",
            label="Unit Circle",
        )
        ax.scatter(
            np.real(eigenvalues),
            np.imag(eigenvalues),
            c=np.arange(len(eigenvalues)),
            cmap="viridis",
            s=60,
            zorder=3,
            label="Eigenvalues",
        )
        ax.axhline(0.0, color="lightgray", linewidth=0.8)
        ax.axvline(0.0, color="lightgray", linewidth=0.8)
        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax.set_title(f"Transfer Operator Spectrum (lag={self.lag})")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        plt.tight_layout()
        self.log_image(fig, "eigenvalue_spectrum")
        plt.close(fig)

    def _plot_implied_timescales(self, its: np.ndarray) -> None:
        """
        Plot a bar chart of the leading implied relaxation timescales.

        Parameters
        ----------
        its : numpy.ndarray
            1-D array of implied timescales.
        """
        n_show = min(int(self.n_modes), len(its))
        if n_show == 0:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        modes = np.arange(1, n_show + 1)
        ax.bar(modes, its[:n_show], color="steelblue", edgecolor="black")
        ax.set_xticks(modes)
        ax.set_xlabel("Relaxation Mode Index")
        ax.set_ylabel("Implied Timescale (frames)")
        ax.set_title(f"Implied Timescales (lag={self.lag})")
        plt.tight_layout()
        self.log_image(fig, "implied_timescales")
        plt.close(fig)

    def _plot_its_vs_lags(self, trajectory: np.ndarray, lags: List[int]) -> None:
        """
        Plot implied relaxation timescales as a function of lag time $\tau$.

        Parameters
        ----------
        trajectory : numpy.ndarray
            2-D trajectory array of shape ``(T, D)``.
        lags : list of int
            Sequence of lag times to evaluate.
        """
        valid_lags = [int(l) for l in lags if 0 < int(l) < len(trajectory)]
        if not valid_lags:
            return
        its_per_lag = []
        for l_val in valid_lags:
            _, _, its_l = self._estimate_spectrum(trajectory, lag=l_val)
            its_per_lag.append(its_l)
        its_mat = np.array(its_per_lag)
        if its_mat.ndim != 2 or its_mat.shape[1] == 0:
            return

        n_show = min(int(self.n_modes), its_mat.shape[1])
        fig, ax = plt.subplots(figsize=(7, 5))
        for m_idx in range(n_show):
            ax.plot(
                valid_lags,
                its_mat[:, m_idx],
                marker="o",
                label=f"Mode {m_idx + 1}",
            )
        ax.set_xlabel(r"Lag Time $\tau$ (frames)")
        ax.set_ylabel("Implied Timescale (frames)")
        ax.set_title("Implied Timescales vs. Lag Time")
        ax.legend()
        plt.tight_layout()
        self.log_image(fig, "implied_timescales_vs_lag")
        plt.close(fig)

    def plot(
        self,
        data: Any,
        latent: Any,
        pred: Any,
        labels: Any,
        meta: Any,
    ) -> None:
        """
        Estimate the transfer operator on the accumulated prediction trajectory
        (falling back to ``latent`` if ``pred`` is ``None``), save eigenvalues,
        eigenvectors, and implied timescales, and generate spectrum plots.

        Parameters
        ----------
        data : Any
            Accumulated inputs (unused).
        latent : Any
            Accumulated latent representations.
        pred : Any
            Accumulated latent predictions.
        labels : Any
            Accumulated ground-truth labels (unused).
        meta : Any
            Accumulated metadata (unused).
        """
        trajectory = self._prepare_trajectory(pred if pred is not None else latent)
        if trajectory is None or len(trajectory) <= int(self.lag):
            self.log_warn(
                "TransferOperatorPlotter: insufficient trajectory frames to estimate "
                "transfer matrix."
            )
            return

        eigenvalues, eigenvectors, its = self._estimate_spectrum(
            trajectory, lag=int(self.lag)
        )

        self._save_data(eigenvalues, "eigenvalues")
        self._save_data(eigenvectors, "eigenvectors")
        self._save_data(its, "implied_timescales")

        for idx, val in enumerate(its[: int(self.n_modes)]):
            if np.isfinite(val):
                self.set_metric(f"its_mode_{idx + 1}", float(val))
        if len(eigenvalues) >= 2:
            gap = float(np.abs(eigenvalues[0]) - np.abs(eigenvalues[1]))
            self.set_metric("spectral_gap", gap)

        self._plot_eigenvalue_spectrum(eigenvalues)
        self._plot_implied_timescales(its)
        if self.lags:
            self._plot_its_vs_lags(trajectory, list(self.lags))
