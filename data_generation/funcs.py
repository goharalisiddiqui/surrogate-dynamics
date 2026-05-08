import sys
import os

from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "collective_encoder"))

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Ramachandran

from collective_encoder.utils import compute_mfpt_matrix

from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize


def calculate_dihedrals(run_folder):
    traj_file = os.path.join(run_folder, 'md.xtc')
    # tpr_file = os.path.join(run_folder, 'md.tpr')
    tpr_file = os.path.join(run_folder, '../initialization', 'tpr_initial.tpr')
    u = mda.Universe(tpr_file, traj_file)
    ala = u.select_atoms('resname ALA')
    Rama = Ramachandran(ala).run(step=10, verbose=True)
    phi = Rama.angles[:, 0, 0] / 180 * np.pi
    psi = Rama.angles[:, 0, 1] / 180 * np.pi
    
    return phi, psi

def population_plot(psi, phi, save_path, save_fig=False):
    fig, ax = plt.subplots(2, 1, figsize=(5,5))
    ax[0].scatter(range(len(phi)), phi, marker='.', s=1, alpha=0.1)
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel(r"$\Phi$")
    ax[0].set_ylim([-np.pi, np.pi])
    ax[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax[0].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    ax[1].scatter(range(len(psi)), psi, marker='.', s=1, alpha=0.1)
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel(r"$\Psi$")
    ax[1].set_ylim([-np.pi, np.pi])
    ax[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_path, dpi=300)
    
def rama_plot(psi, phi, save_path, save_fig=False):
    fig, ax = plt.subplots(1, 1, figsize=(5,5))

    ax.scatter(phi, psi, s=1)
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    fig.tight_layout()
    if save_fig:
        fig.savefig(save_path, dpi=300)
        
'''
1) Use a standard Gaussian kernel density estimator (Scott, 1992) to approximate the free energy surface in the space of the two dihedral angles ϕ, ψ 
2) Run 100 BFGS solvers (Nocedal & Wright, 2006) initialized at random points and run until convergence from which we recover the unique local minima. 

By doing so, we are able to reliably identify metastable states without the need for manual specification
'''

def periodic_norm_distance(x1, x2, period=2*np.pi):
    """ Compute the periodic norm distance between two points x1 and x2 """
    # Minimum image convention
    d = (x1 - x2 + period/2) % period - period/2
    
    return np.linalg.norm(d)

def periodic_average(x, period=2*np.pi):
    """ Compute the periodic average of a list of 2D points x """
    # Minimum image convention
    x = np.array(x)
    x1 = np.arctan2(np.mean(np.sin(x[:,0])), np.mean(np.cos(x[:,0])))
    x2 = np.arctan2(np.mean(np.sin(x[:,1])), np.mean(np.cos(x[:,1])))
    return np.array([x1, x2])

def run_bfgs(phi, psi, 
             sigma=0.2, # bandwidth for KDE
             bin_x=200, bin_y=200, 
             minima_tolerance=0.15, 
             n_BFGS_runs=100):
    
    # Step 1
    X = np.stack([phi, psi], axis=-1)
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X) # bandwidth is sigma

    # Step 2
    def objective(x):
        return -kde.score_samples(x.reshape(1, -1))
    results = []
    for _ in tqdm(range(n_BFGS_runs), desc="Running BFGS optimizations"):
        x0 = np.random.uniform(-np.pi, np.pi, size=2)
        res = minimize(objective, x0, method='BFGS')
        results.append(res)
    min_points = np.array([r.x for r in results if r.success])
    # Remove points whole norm are closer than 'minima_tolerance'
    min_points_reduced = []
    for p in min_points:
        for q in min_points_reduced:
            if any(periodic_norm_distance(np.array(p), np.array(r)) <= minima_tolerance for r in q):
                q.append(p)
                break
        else:
            min_points_reduced.append([p])
    min_points = np.array([periodic_average(a) for a in min_points_reduced])
    print("Number of local minima found:", len(min_points))
    print(min_points)
    return kde, min_points

def compute_fes(kde, bin_x=200, bin_y=200):
    x = np.linspace(-np.pi, np.pi, bin_x) # grid points
    y = np.linspace(-np.pi, np.pi, bin_y)
    Xgrid, Ygrid = np.meshgrid(x, y)
    # Free energy surface
    kbt = 0.415 # at 300K in kJ/mol
    fes = kbt * -kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T).reshape(Xgrid.shape)
    fes -= fes.min() # set minimum to zero
    
    return Xgrid, Ygrid, fes

def sort_minima_by_depth(kde, min_points):
    min_vals = np.array([float(-kde.score_samples(p.reshape(1, -1))[0]) for p in min_points])
    print("Objective values at minima:", min_vals)
    # Sorting minima based on their depth
    sorted_indices = np.argsort(min_vals)
    min_points = min_points[sorted_indices]
    min_vals = min_vals[sorted_indices]
    print("Sorted objective values at minima:\n", min_vals)
    print("Sorted minima points:\n", min_points)
    
    return min_points

def plot_fes(Xgrid, Ygrid, fes, min_points, save_path = None):
    """
    Plot the free energy surface (FES) with color limits fixed to [0.0, 8.0]
    and colorbar ticks from 0.0 to 8.0 with step 1.0.
    """
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))

    vmin, vmax = 0.0, fes.max()
    # Clip FES to plotting range so colors are bounded
    fes_clipped = np.clip(fes, vmin, vmax)

    # Use levels spanning the fixed range for consistent contour steps
    levels = np.linspace(vmin, vmax, 30)
    cf = ax.contourf(Xgrid, Ygrid, fes_clipped, levels=levels, cmap=cm.viridis, vmin=vmin, vmax=vmax)

    ccb = fig.colorbar(cf, ax=ax, label='Free Energy (kJ/mol)')
    ticks = np.arange(vmin, vmax + 1e-9, 1.0)  # 0.0..8.0 step 1.0
    ccb.set_ticks(ticks)
    ccb.set_ticklabels([f"{t:.1f}" for t in ticks])

    ax.scatter(min_points[:,0], min_points[:,1], facecolors='w', edgecolors='r', s=200, label='Local Minima')
    for i, (x, y) in enumerate(min_points):
        ax.text(x, y, str(i + 1), color='black', fontsize=8, ha='center', va='center')

    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel(r"$\Psi$")
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    # ax.legend(bbox_to_anchor=(1, 1.1))

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)

def calculate_mfpt(phi, psi, min_points, lag_time=1):
    minima = min_points
    msm_data = np.stack([phi, psi], axis=-1)
    # n_max_states = 4
    # if len(minima) > n_max_states:
    #     minima = minima[:n_max_states]
    transition_counts, transition_matrix, mfpt_matrix = compute_mfpt_matrix(msm_data, minima, lag=lag_time)
    
    return minima, transition_counts, transition_matrix, mfpt_matrix

def plot_mfpt_matrix(minima,
                     mfpt_matrix, 
                     transition_counts,
                     transition_matrix, 
                     lag_time = 1,
                     save_path = None):
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    # Plot MFPT matrix
    ax.matshow(mfpt_matrix, cmap='viridis')
    mfpt_min = mfpt_matrix.min()
    # fig.colorbar(cm.ScalarMappable(cmap='viridis', 
    #                             norm=plt.Normalize(vmin=mfpt_min if mfpt_min > 0 else 1e-9, vmax=mfpt_matrix.max())), 
    #                             ax=ax, 
    #                             label='MFPT (steps)')
    # ax.set_title(f'MFPT Matrix (lag_time={lag_time})')
    ax.set_xlabel('Target State')
    ax.set_ylabel('Source State')
    ax.set_xticks(np.arange(len(minima)))
    ax.set_yticks(np.arange(len(minima)))
    ax.set_xticklabels([str(i) for i in range(1, len(minima) + 1)])
    ax.set_yticklabels([str(i) for i in range(1, len(minima) + 1)])
    fig.tight_layout()
    if save_path:
        fig.savefig(os.path.join(save_path, 'mfpt_matrix.png'), dpi=300)
    plt.close()

    # Plot Transition counts 
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    tm_norm = LogNorm(vmin=0.00102, vmax=1.0)
    # transition_matrix = np.log1p(transition_matrix)
    ax.matshow(transition_matrix, cmap='viridis', norm=tm_norm)
    # cb = fig.colorbar(cm.ScalarMappable(cmap='viridis', 
    #                             norm=tm_norm), 
    #                             ax=ax, 
    #                             label='Transition Probabilities')
    # cb.set_ticks([1.0, 0.1, 0.01, 0.001])
    # cb.set_ticklabels(['1', '0.1', '0.01', '0.001'])
    # cb.set_label('Transition Probabilities (log scale)', fontsize=14)
    # small colorbar
    # ax.set_title(f'Transition Matrix (lag_time={lag_time})')
    ax.set_xlabel('To State', fontsize=14)
    ax.set_ylabel('From State', fontsize=14)
    ax.set_xticks(np.arange(len(minima)))
    ax.set_yticks(np.arange(len(minima)))
    ax.set_xticklabels([str(i) for i in range(1, len(minima) + 1)])
    ax.set_yticklabels([str(i) for i in range(1, len(minima) + 1)])

    fig.tight_layout()


    print(f"  Shape: {mfpt_matrix.shape}")
    print(f"  Lag time: {lag_time}")
    
    if save_path:
        fig.savefig(os.path.join(save_path, 'transition_matrix.png'), dpi=300)