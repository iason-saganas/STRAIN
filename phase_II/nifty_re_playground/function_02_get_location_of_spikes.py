from useful.calculate_pseudoinverse import find_penrose_moore_solution, sample_from_ps
from phase_II.nifty_re_playground.useful.helpers import *

def _02_get_location_of_spikes_from_xi(pipe:InferenceSchemeRe, output_folder:str, harmonic_xi:jnp.array=None,
                                       penrose_xi_iter=100_000):
    """
    This function retrieves the peak locations of a xi variable. It's intended use is in the iterative refinement of the 
    noise model through additional Gaussian combs.
     
    :param pipe:             The InferenceSchemeRe object describing the current run.
    :param output_folder:    The out folder in which meta-data (figures of peak locations) are stored
    :param harmonic_xi:      The harmonic xi array to search peakes in. If none, the penrose xi is computed and
                             stored instead.
    :return:
    """
    output_folder = output_folder + f"/peak_finder/"
    plots_exist = os.path.isdir(output_folder)
    if not plots_exist:
        # Create folder
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    data_grid = pipe.d_dom_real
    harmonic_signal_grid = pipe.s_dom_harmonic
    signal_distributor = harmonic_signal_grid.power_distributor
    
    posterior_pipe_ps_mean_std, _, _ = pipe.get_posterior_statistics()
    posterior_pipe_ps_mean = (posterior_pipe_ps_mean_std[0])[signal_distributor]
    
    if harmonic_xi is None:
        print("Recalculating penrose moore xi, since harmonic_xi is None")
        fn = output_folder + "penrose_xi.txt"
        harmonic_xi = find_penrose_moore_solution(pipe=pipe, itr=penrose_xi_iter, reload_from_cache=True,
                                                  filename=fn)
    else:
        fn = output_folder + "harmonic_xi.txt"
        f = pipe.k_signal
        to_save = np.column_stack((harmonic_xi, pipe.k_signal_full))
        np.savetxt(fn, to_save, )

    peaks_k, norm_amplitudes_k = get_peaks_from_cache_v2(local_sigma_threshold=3, global_sigma_threshold=2,
                                                         window_length=20, take_abs_of_amplitudes=True,
                                                         custom_path=fn)

    if not plots_exist:

        # Plot of harmonic_xi and its 2 sigma peaks
        plt.plot(pipe.k_signal_full, harmonic_xi.real, color=red)
        plt.plot(peaks_k, [0.1*max(harmonic_xi.real)]*len(peaks_k), "b.", markersize=5, label="Detected peaks")
        usual_plot(xl="Frequency $f$", yl="Arbitrary units", title=r"$\tilde{\xi}_d$", show=False, save_fig=True,
                   save_path=output_folder+"/xi_with_peaks", close=True,)


        # Plot of smooth background together with found peaks
        where_positive = np.where(pipe.k_signal_full>0)
        plt.plot(pipe.k_signal_full[where_positive], posterior_pipe_ps_mean[where_positive],
                 color=blue, label=r"Smooth background $p_n(f)$")
        plt.plot(peaks_k, norm_amplitudes_k, color=red, marker="v", markersize=5, linewidth=0,
                 label=r"Peaks found in $\tilde{\xi}_d$")
        plot_welch_averaged_ps()
        plt.loglog()
        usual_plot(xl="Frequency $f$", yl="Power", show=False, save_fig=True, close=True,
                   save_path=output_folder+"/power_spectrum_with_peaks",)


        # Plot in data space
        ps_mean_std, _, _ = pipe.get_posterior_statistics()
        ps_mean = (ps_mean_std[0])[pipe.s_h_dom_expander]
        posterior_penrose_data = sample_from_ps(harmonic_xi, N=pipe.n_ds, inverse_h_trafo=lambda p: jnp.fft.ifft(p, norm="ortho"),
                                                ps=ps_mean)

        plt.plot(pipe.t_ds, pipe.d, label=r"$d_{\mathrm{obs}}$", color="orange")
        plt.plot(pipe.t_ds, posterior_penrose_data.real, color=blue, label=r"Data from smooth $p_n(f)$ and $\tilde{\xi}_d$")
        usual_plot(save_fig=True, show=False, save_path=output_folder+"/data_comparison_if_penrose_xi", close=True,)

    return peaks_k, norm_amplitudes_k

