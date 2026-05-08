from funcs import *

sigma = 0.2 # bandwidth for KDE
bin_x = 200
bin_y = 200
minima_tolerance = 0.15
n_BFGS_runs = 200
lag_time = 1


run_folders = [
    # 'ala2/solvated/200K/ala2_100ns',
    # 'ala2/solvated/273K/ala2_100ns',
    # 'ala2/solvated/283K/ala2_100ns',
    # 'ala2/solvated/293K/ala2_100ns',
    'ala2/solvated/300K/ala2_100ns',
    # 'ala2/solvated/303K/ala2_100ns',
    # 'ala2/solvated/313K/ala2_100ns',
    # 'ala2/solvated/323K/ala2_100ns',
    # 'ala2/solvated/333K/ala2_100ns',
    # 'ala2/solvated/343K/ala2_100ns',
    # 'ala2/solvated/353K/ala2_100ns',
    # 'ala2/solvated/363K/ala2_100ns',
    # 'ala2/solvated/373K/ala2_100ns',
]
for run_folder in run_folders:
    phi, psi = calculate_dihedrals(run_folder)
    # save_path = os.path.join(run_folder, '..','population_plot.png')
    # population_plot(psi, phi, save_path, save_fig=True)
    # save_path = os.path.join(run_folder, '..','rama_plot.png')
    # rama_plot(psi, phi, save_path, save_fig=True)
    kde, min_points = run_bfgs(phi, psi, 
                            sigma, bin_x, bin_y, 
                            minima_tolerance, n_BFGS_runs)
    min_points = sort_minima_by_depth(kde, min_points)
    selection = [1,2,3,4,6,7]#list(range(min(6, len(min_points))))
    min_points = min_points[selection]
    Xgrid, Ygrid, fes = compute_fes(kde, bin_x, bin_y)
    save_path = os.path.join(run_folder, '..', 'plot_fes.png')
    plot_fes(Xgrid, Ygrid, fes, min_points, save_path=save_path)
    # states 
    minima, transition_counts, transition_matrix, mfpt_matrix = calculate_mfpt(phi, psi, min_points, lag_time=lag_time)
    plot_mfpt_matrix(minima,
                     mfpt_matrix,
                     transition_counts,
                     transition_matrix,
                     save_path=os.path.join(run_folder,'..'),
                     lag_time=lag_time)
    with open(os.path.join(run_folder,'..', 'minima.txt'), 'w') as f:
        for i, p in enumerate(min_points):
            f.write(f"{p[0]:.4f}, {p[1]:.4f}\n")