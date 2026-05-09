import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib import cm

from utils import (
    periodic_norm_distance,
    periodic_average
)

def build_kde(phi, psi, sigma):
    X = np.stack([phi, psi], axis=-1)
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X)
    return kde

def plot_kde(kde, bin_x, bin_y):
    # Create grid for evaluation
    x = np.linspace(-np.pi, np.pi, bin_x) # grid points
    y = np.linspace(-np.pi, np.pi, bin_y)
    Xgrid, Ygrid = np.meshgrid(x, y)

    # Free energy surface
    kbt = 0.415 # at 300K in kJ/mol
    fes = kbt * -kde.score_samples(
                    np.vstack(
                        [Xgrid.ravel(), Ygrid.ravel()]
                        ).T).reshape(Xgrid.shape)
    fes -= fes.min() # set minimum to zero

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    cf = ax.contourf(Xgrid, Ygrid, fes, levels=30, cmap=cm.viridis)
    fig.colorbar(cf, ax=ax, label='Free Energy (kJ/mol)')
    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel(r"$\Psi$")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    plt.tight_layout()
    
    return fig, ax

def build_msm(kde, n_BFGS_runs, minima_tolerance):
    def objective(x):
        return -kde.score_samples(x.reshape(1, -1))

    results = []
    for _ in range(n_BFGS_runs):
        x0 = np.random.uniform(-np.pi, np.pi, size=2)
        res = minimize(objective, x0, method='BFGS')
        results.append(res)
    min_points = np.array([r.x for r in results if r.success])
    # Remove points whole norm are closer than 'minima_tolerance'
    min_points_reduced = []
    for p in min_points:
        for q in min_points_reduced:
            if any(periodic_norm_distance(np.array(p), np.array(r)) <= \
                                                minima_tolerance for r in q):
                q.append(p)
                break
        else:
            min_points_reduced.append([p])
    min_points = np.array([periodic_average(a) for a in min_points_reduced])
    min_vals = np.array([float(objective(p)[0]) for p in min_points])
    sorted_indices = np.argsort(min_vals)
    min_points = min_points[sorted_indices]
    min_vals = min_vals[sorted_indices]

    return min_points, min_vals

def check_correct_minima(minima_model, minima_ref, tolerance):
    minima_model = minima_model[:len(minima_ref)]
    dists, correct_dists, idx_correct  = [], [], []
    for i in range(len(minima_model)):
        dist = periodic_norm_distance(minima_model[i], minima_ref[i], period=2*np.pi)
        dists.append(dist)
        if dist <= tolerance:
            correct_dists.append(dist)
            idx_correct.append(i)

    return minima_model[idx_correct], correct_dists, dists
