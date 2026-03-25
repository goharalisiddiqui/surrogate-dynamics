from funcs import *

sigma = 0.2 # bandwidth for KDE
bin_x = 200
bin_y = 200
minima_tolerance = 0.15
n_BFGS_runs = 100
lag_time = 1


run_folders = [
    'ala2/solvated/200K/ala2_100ns',
]
for run_folder in run_folders:
    phi, psi = calculate_dihedrals(run_folder)
    save_path = os.path.join(run_folder, '..','population_plot.png')
    population_plot(psi, phi, save_path, save_fig=True)
    save_path = os.path.join(run_folder, '..','rama_plot.png')
    rama_plot(psi, phi, save_path, save_fig=True)
    kde, min_points = run_bfgs(phi, psi, 
                            sigma, bin_x, bin_y, 
                            minima_tolerance, n_BFGS_runs)
    min_points = sort_minima_by_depth(kde, min_points)
    Xgrid, Ygrid, fes = compute_fes(kde, bin_x, bin_y)
    save_path = os.path.join(run_folder, '..', 'plot_fes.png')
    plot_fes(Xgrid, Ygrid, fes, min_points, save_path=save_path, save_fig=True)
    # states 
    selected = [a for a in range(len(min_points))] # choose which states to consider
    minima, transition_counts, transition_matrix, mfpt_matrix = calculate_mfpt(phi, psi, min_points, selected, lag_time)
    # plot_mfpt_matrix(minima,
    #                  mfpt_matrix,
    #                  transition_counts,
    #                  transition_matrix,
    #                  selected,
    #                  save_path=os.path.join(run_folder,'..', 'mfpt_matrix.png'),
    #                  lag_time=lag_time)