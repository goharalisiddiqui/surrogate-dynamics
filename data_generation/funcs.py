import sys
import os
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
    for _ in range(n_BFGS_runs):
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

def plot_fes(Xgrid, Ygrid, fes, min_points, save_path, save_fig=False):
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
    cf = ax.contourf(Xgrid, Ygrid, fes, levels=30, cmap=cm.viridis)
    fig.colorbar(cf, ax=ax, label='Free Energy (kJ/mol)')
    ax.scatter(min_points[:,0], min_points[:,1], 
            facecolors='w', edgecolors='r', s=200, 
            label='Local Minima')
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
    ax.legend(bbox_to_anchor=(1, 1.1))

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_path, dpi=300)

def calculate_mfpt(phi, psi, min_points, selected, lag_time=1):
    selected_idx = [i-1 for i in selected]
    print("Selected states:", min_points[selected_idx])
    minima = min_points[selected_idx]
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
                     selected, 
                     save_path,
                     lag_time = 1,
                     save_fig=False):
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    # Plot MFPT matrix
    ax[0].matshow(mfpt_matrix, cmap='viridis')
    fig.colorbar(cm.ScalarMappable(cmap='viridis', 
                                norm=plt.Normalize(vmin=mfpt_matrix.min(), vmax=mfpt_matrix.max())), 
                                ax=ax[0], 
                                label='MFPT')
    ax[0].set_title(f'MFPT Matrix (lag_time={lag_time})')
    ax[0].set_xlabel('Target State')
    ax[0].set_ylabel('Source State')
    ax[0].set_xticks(np.arange(len(minima)))
    ax[0].set_yticks(np.arange(len(minima)))
    ax[0].set_xticklabels([str(i) for i in selected])
    ax[0].set_yticklabels([str(i) for i in selected])


    # Plot Transition counts 
    print(transition_matrix.min(), transition_matrix.max())
    tm_norm = LogNorm(vmin=transition_matrix.min(), vmax=transition_matrix.max())
    # transition_matrix = np.log1p(transition_matrix)
    ax[1].matshow(transition_matrix, cmap='viridis', norm=tm_norm)
    fig.colorbar(cm.ScalarMappable(cmap='viridis', 
                                norm=tm_norm), 
                                ax=ax[1], 
                                label='Transition Probabilities')
    ax[1].set_title(f'Transition Matrix (lag_time={lag_time})')
    ax[1].set_xlabel('To State')
    ax[1].set_ylabel('From State')
    ax[1].set_xticks(np.arange(len(minima)))
    ax[1].set_yticks(np.arange(len(minima)))
    ax[1].set_xticklabels([str(i) for i in selected])
    ax[1].set_yticklabels([str(i) for i in selected])

    fig.tight_layout()


    print(f"  Shape: {mfpt_matrix.shape}")
    print(f"  Lag time: {lag_time}")
    print(f"  Transition counts:\n{transition_counts}")
    print(f"  MFPT Matrix:\n{mfpt_matrix}")
    
    if save_fig:
        fig.savefig(save_path, dpi=300)