import matplotlib.pyplot as plt

# Good run: Set plot_prior_samples to False.
from _01_get_smooth_baseline_ps_debug import *
from function_02_get_location_of_spikes import _02_get_location_of_spikes_from_xi
import numpy as np
import nifty.nifty.re as jft

if __name__ == "__main__":
    pipe_2_called_as_import = False
else:
    pipe_2_called_as_import = True


global_peak_refinement_steps = 6
pipe = pipe_1
h_xi = None  # First, we need to find a xi via penrose minimization that fits the data well
switch_to_inference_xi_in_iter = [0, 1, 1, 1, 1, 1, None]  # if 1, h_xi gets updated to inference xi, otherwise penrose xi is recomputed
# Also put None for the last step!
penrose_minimization_iter = [100_000, 100_000, 100_000, 100_000, 100_000, 100_000]
kl_iter = [5, 5, 5, 5, 10, 10]
peak_posterior_metadata = []  # positions and amplitudes
latent_peak_posterior_metadata = []  # positions and latent amplitude values
old_pipes = []  # all inference schemes

for r in range(global_peak_refinement_steps):

    # 0. Initialize inference scheme
    pipe_2 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key, plotting_callback=analyze_kl_callback)
    out_name = "re_pipe_2_debug"
    sub_out_name = out_name + f"/peak_refinement_{r}"

    # 1. Optional: Get some kind of template operator to 'regularize' problem
    prior_ps_template_mean_std, _, return_posterior_parameters = pipe_1.get_posterior_statistics()
    prior_ps_template_mean = prior_ps_template_mean_std[0]
    template_operator = ScaledPowerSpectrumTemplate(ps_template=prior_ps_template_mean, scale=(1, .1))

    # 2. Get amplitudes and positions of peaks in penrose xi or inference harmonic xi to build line model operator
    # Make sure sigma threshhold is consistent with '_02_' file
    try:
        peak_pos, peak_amps = _02_get_location_of_spikes_from_xi(pipe=pipe, output_folder=sub_out_name, harmonic_xi=h_xi,
                                                                 penrose_xi_iter=penrose_minimization_iter[r])

    except ValueError as e:
        print("Error occured: ", e, " => falling back to previous inferred xi value for peak estimation")
        h_xi = old_pipes[r-1].plot_posterior_harmonic_xi_s(show=False)
        peak_pos, peak_amps = _02_get_location_of_spikes_from_xi(pipe=pipe, output_folder=sub_out_name,
                                                                 harmonic_xi=h_xi,
                                                                 penrose_xi_iter=penrose_minimization_iter[r])

    # Check out penrose xi GW candidates
    # if r != 0 and switch_to_inference_xi_in_iter[r] == 0:
    #     # Penrose xi was successfully calculated, get analyze it
    #     print("Calculating (or catching) penrose xi in iteration ", r)
    #     obj = np.loadtxt(sub_out_name + "/peak_finder/penrose_xi.txt", dtype=np.complex128)
    #     penrose_xi, _ = (obj[:, 0]).real, obj[:, 1].real
    #
    #     _ = plt.figure(figsize=(8.,4.))
    #     plt.plot(pipe_2.k_signal_full, penrose_xi.real, color=blue)
    #     thesis_plot(mode="longer", xl=r"Frequency $\mathrm{[Hz]}$", yl="Amplitude (arb. units)", save_fig=True)
    #
    #     S_penrose, t_dual, f_dual = Stress_jft(penrose_xi, time=time, downsample=False)
    #     # pickle_me_this("wigner_results_from_penrose_xi_without_welch_average", [S, t_dual, f_dual])
    #     # S, t_dual, f_dual = unpickle_me_this("wigner_results_from_penrose_xi_without_welch_average.pickle")
    #
    #     plt.figure(figsize=(6, 6))
    #     ax = plt.gca()
    #     cb, im = visualize_stress(S_penrose, f_dual, t_dual, smooth=True, smoothing_level=3,
    #                               save_fig=False, xlim=(min(time), max(time)), cmap="plasma", delay_plot=True,
    #                               custom_ax=ax, return_aux=True)
    #     plt.xlim(16.3, 16.5)
    #     plt.ylim(-400, 400)
    #     plt.xlabel(r"Time $\mathrm{[s]}$")
    #     plt.ylabel(r"Frequency $\mathrm{[Hz]}$")
    #     plt.subplots_adjust(right=0.8, left=0.2)
    #     save_figure(save_fig=False, tight_ly=False, show=True)

    # Unpack peak information from this and earlier runs
    earlier_peak_positions = np.array([tpl[0] for tpl in peak_posterior_metadata])
    earlier_peak_amplitudes = np.array([tpl[1] for tpl in peak_posterior_metadata])

    if earlier_peak_positions.size > 0:
        # safely append
        peak_pos = np.concatenate((peak_pos, earlier_peak_positions))
        peak_amps = np.concatenate((peak_amps, earlier_peak_amplitudes))

    relative_std_in_peak_amplitude = 1
    # relative_std_in_peak_amplitude = 1e-16
    relative_std_in_peak_width = 1

    gaussian_comb_op = BaselineNormedGaussianComb(
        unique_k_lengths=pipe_1.k_signal,
        list_of_peaks=peak_pos,
        list_of_amplitudes_above_baseline=np.array([1e1]*len(peak_amps)),
        a_priori_width_of_peaks=1,
        rel_sigma_widths=relative_std_in_peak_width,
        rel_sigma_amp=relative_std_in_peak_amplitude,
        norm=False,
    )

    # plot_histogram(key=key, mean=1e1, sigma=1e5, n_samples=5000, mode="Lognormal")

    # 4. Build correlated field model with custom operators
    deviation_correlated_field = {"fluct": (2,2), "llslope": (-2,1), "flex":(1,1), "square_iwp": False}  # with template operator
    # deviation_correlated_field = {"fluct": (3,2), "llslope": (-2,1), "flex":(.5,.1), "square_iwp": False,
    #                               "apply_tukey_window": False}  # without
    # where 1 is from the normed Gaussian comb
    pipe_2.add_cfm_signal_model(add_power_spectrum_template=template_operator, add_peak_model=gaussian_comb_op,
                                **deviation_correlated_field)

    # 5. Add noise level, here, I learn the noise level because I actually don't know
    # N_cov = jft.LogNormalPrior(mean=.1, std=.1, name="additional_noise_var", dtype=jnp.float64, shape=())
    # N_inv_cov = jft.Model(domain=N_cov.domain, call=lambda xi: N_cov(xi)**(-1))
    # N_inv_std_cov = jft.Model(domain=N_cov.domain, call=lambda xi: jnp.sqrt(N_inv_cov(xi)))
    # pipe_2.add_noise_op(inverse_noise_op=N_inv_cov, sqrt_inverse_noise_op=N_inv_std_cov)
    data_noise_level_coarse = 1e-3
    pipe_2.add_noise_op(noise_var_level=data_noise_level_coarse)

    # 6. Set initial position of peaks from earlier runs
    if not pipe_2_called_as_import and r != 0:

        op = gaussian_comb_op.xi_g_amp
        key, key_i = jax.random.split(key)

        current_r_peak_positions = gaussian_comb_op.positions  #  If you allow peak positions to vary, this will break
        random_latent_amplitude_draw = jft.random_like(key=key_i, primals=op.domain)

        earlier_peak_positions = np.array([tpl[0] for tpl in peak_posterior_metadata])
        earlier_peak_latent_amplitudes = np.array([tpl[1] for tpl in latent_peak_posterior_metadata])

        # 6.1. Ensure all earlier peaks exist in current peaks
        if not np.all(np.isin(earlier_peak_positions, current_r_peak_positions)):
            missing = earlier_peak_positions[~np.isin(earlier_peak_positions, current_r_peak_positions)]
            raise ValueError(f"Some earlier peaks not in current peaks: {missing}")
        # 6.2. Find the indices in current_r_peak_positions corresponding to earlier_peak_positions
        indices = np.array([np.where(current_r_peak_positions == pos)[0][0] for pos in earlier_peak_positions])
        # 6.3. Create a new array for latent amplitudes (jax arrays are immutable)
        latent_amplitude_as_array = random_latent_amplitude_draw["xi_g_amp"]
        latent_amplitude_init = latent_amplitude_as_array.at[indices].set(earlier_peak_latent_amplitudes)

        debug = False
        if debug:
            i = -1
            for idx in indices:
                i += 1
                print(earlier_peak_positions[i], current_r_peak_positions[idx], " --> ", latent_amplitude_init[idx],
                      earlier_peak_latent_amplitudes[i])

        pipe_2.set_init_pos(init_pos={"xi_g_amp": latent_amplitude_init}, plot=False)

    plot_prior_samples = False
    if not pipe_2_called_as_import and plot_prior_samples:

        # plt.plot(peak_pos, peak_amps, "b.", markersize=3, label="Input peaks")
        # plt.plot(pipe_1.k_signal, prior_ps_template_mean, label="Prior template")
        # plt.loglog()
        # plt.show()

        pipe_2.plot_prior_samples(mode="power spectrum", num=5, plot_welch_average=True, rolling=False, plot_data=False)
        pipe_2.plot_prior_samples(mode="signal", num=2, rolling=False, plot_welch_average=False)

        pipe_2.plot_prior_samples(mode="signal & power spectrum", num=5)

    latent_samples, vi_info = pipe_2.run_inference(kl_iterations=kl_iter[r], use_strict_minimizers=False,
                                          out_name=f"{sub_out_name}", resume=True)
    key = pipe_2.get_current_key()
    pipe = pipe_2  # bump from pipe_1 to pipe_2 in the iterative refinement
    if switch_to_inference_xi_in_iter[r+1] == 1:
        h_xi = pipe_2.plot_posterior_harmonic_xi_s(only_return=True)  # now we can check for peaks in this xi in the next
        # refinement iteration
    else:
        h_xi = None # recompute penrose

    print("FINISHED PEAK REFINEMENT ITERATION ", r)

    # Save posterior gaussian comb for initilization in the next refinement iteration
    posterior_peak_positions = gaussian_comb_op.positions
    posterior_peak_amplitudes = jft.mean([gaussian_comb_op.xi_g_amp(xi) for xi in latent_samples])
    posterior_peak_latent_amplitudes = jft.mean([xi["xi_g_amp"] for xi in latent_samples])
    posterior_peak_info = list(zip(posterior_peak_positions, posterior_peak_amplitudes))
    posterior_latent_peak_info = list(zip(posterior_peak_positions, posterior_peak_latent_amplitudes))
    peak_posterior_metadata = posterior_peak_info
    latent_peak_posterior_metadata = posterior_latent_peak_info
    old_pipes.append(pipe)

    thesis_plot_3 = True
    plot_results = False
    if not pipe_2_called_as_import and plot_results:

        pipe_2.plot_posterior_signal()
        save_fig = False
        # if r == 5:
        #     save_fig = True
        pipe_2.plot_posterior_power_spectrum(mode="mean", save_fig=save_fig)
        pipe_2.plot_posterior_harmonic_xi_s()

        # pipe_2.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)

        # xi_s_harmonic = pipe_2.plot_posterior_harmonic_xi_s(show=False)
        # S, t_dual, f_dual = Stress_re(xi_s_harmonic, time=time, downsample=False)
        # visualize_stress(S, f_dual, t_dual, smooth=True, detect_outliers=False)

        if r == 5:
            print("Calculating (or catching) penrose xi in iteration ", r)
            penrose_xi = pipe_2.calculate_and_plot_penrose_xi(itr=100_000, plot=True, reload_from_cache=True, fn=f"penrose_xi_in_iter_{r}.txt")

            # _ = plt.figure(figsize=(8.,4.))
            # plt.plot(pipe_2.k_signal_full, penrose_xi.real, color=blue)
            # thesis_plot(mode="longer", xl=r"Frequency $\mathrm{[Hz]}$", yl="Amplitude (arb. units)", save_fig=True)

            S_penrose, t_dual, f_dual = Stress_jft(penrose_xi, time=time, downsample=False)
            # pickle_me_this("wigner_results_from_penrose_xi_without_welch_average", [S, t_dual, f_dual])
            # S, t_dual, f_dual = unpickle_me_this("wigner_results_from_penrose_xi_without_welch_average.pickle")

            plt.figure(figsize=(6, 6))
            ax = plt.gca()
            cb, im = visualize_stress(S_penrose, f_dual, t_dual, smooth=True, smoothing_level=3,
                                      save_fig=False, xlim=(min(time), max(time)), cmap="plasma", delay_plot=True,
                                      custom_ax=ax, return_aux=True)
            plt.xlim(16.3, 16.5)
            plt.ylim(-400, 400)
            plt.xlabel(r"Time $\mathrm{[s]}$")
            plt.ylabel(r"Frequency $\mathrm{[Hz]}$")
            plt.subplots_adjust(right=0.8, left=0.2)
            save_figure(save_fig=False, tight_ly=False, show=True)
        # if r == 6:
        #     import jax.numpy as jnp
        #     xi_s_harmonic = pipe_2.plot_posterior_harmonic_xi_s(show=False)
        #
        #     # Notch filter
        #     mask = jnp.abs(xi_s_harmonic) > 0.1
            # xi_s_harmonic = xi_s_harmonic.at[mask].set(0.0)
            #
            # plt.plot(pipe_2.k_signal_full, xi_s_harmonic)
            # plt.show()
            #
            # S, t_dual, f_dual = Stress_jft(xi_s_harmonic, time=time, downsample=False)
            # visualize_stress(S, f_dual, t_dual, smooth=True)


    if thesis_plot_3 and r == 5:
        fig = plt.figure(constrained_layout=False, figsize=(10, 8))
        gridspec = fig.add_gridspec(nrows=3, ncols=2)

        # Get Wigner function and store
        penrose_xi = pipe_2.calculate_and_plot_penrose_xi(itr=100_000, plot=False, reload_from_cache=True,
                                                          fn=f"penrose_xi_in_iter_{r}.txt")

        inference_xi = pipe_2.plot_posterior_harmonic_xi_s(only_return=True)

        # S_penrose, _, _ = Stress_jft(penrose_xi, time=time)
        # S_inference, _, _ = Stress_jft(inference_xi, time=time)

        # import os
        # os.makedirs("dlt_later/", exist_ok=True)
        # pickle_me_this("dlt_later/S_penrose", S_penrose)
        # pickle_me_this("dlt_later/S_inference", S_inference)

        S_penrose = unpickle_me_this("dlt_later/S_penrose.pickle")
        S_inference = unpickle_me_this("dlt_later/S_inference.pickle")

        # Create subplot grids

        ax_upper = fig.add_subplot(gridspec[0, :])  # first row, all columns

        ax_11 = fig.add_subplot(gridspec[1, 0])
        ax_12 = fig.add_subplot(gridspec[1, 1])

        ax_21 = fig.add_subplot(gridspec[2, 0])
        ax_22 = fig.add_subplot(gridspec[2, 1])


        #--- Populate: Upper row
        pipe_2.plot_posterior_power_spectrum(mode="mean", custom_ax=ax_upper, plot_welch_average=False,
                                             label="Posterior mean")
        plot_welch_averaged_ps(ax=ax_upper, alpha=0.5)


        # --- Populate: Upper left (under upper row): ax_11
        pipe_2.plot_posterior_harmonic_xi_s(custom_ax=ax_11, custom_xi=inference_xi/np.max(inference_xi),
                                            label="")

        # --- Populate: Lower left: ax_21
        pipe_2.plot_posterior_harmonic_xi_s(custom_ax=ax_21, custom_xi=penrose_xi.real/np.max(penrose_xi.real),
                                            label="")

        cmap = 'seismic'

        # Populate imshow plots
        visualize_stress(stress_matrix=S_inference/np.max(S_inference), rows=pipe_2.k_signal_full,
                         cols=pipe_2.t_ss-16.4, smooth=True, custom_ax=ax_12, delay_plot=True,
                         colorbar_label="Stress (normed)", cmap=cmap)

        visualize_stress(stress_matrix=S_penrose/np.max(S_penrose), rows=pipe_2.k_signal_full,
                         cols=pipe_2.t_ss-16.4, smooth=True, custom_ax=ax_22, delay_plot=True,
                         colorbar_label="Stress (normed)", cmap=cmap)


        # Misc edits
        ax_upper.legend(loc="lower left")
        # ax_11.legend(loc="lower left")
        # ax_21.legend(loc="lower left")

        ax_11.sharex(ax_21)
        ax_11.tick_params(labelbottom=False)

        ax_12.sharex(ax_22)
        ax_12.sharey(ax_22)
        ax_12.tick_params(labelbottom=False)

        ax_11.set_title(r"Posterior mean $\tilde{\xi}_d$")
        ax_21.set_title(r"Moore-Penrose $\tilde{\xi}_d^{\ast}$")

        ax_12.set_xlim(-0.1, 0.1)
        ax_12.set_ylim(-400, 400)

        # Axes labels
        ax_upper.set_ylabel("Power")
        ax_11.set_ylabel("normed Ampl.")
        ax_21.set_ylabel("normed Ampl.")

        ax_21.set_xlabel(r"$f$ $\mathrm{[Hz]}$")

        ax_22.set_xlabel(r"$t$ $\mathrm{[s]}$")
        ax_22.set_ylabel(r"$f$ $\mathrm{[Hz]}$", labelpad=-5)
        ax_12.set_ylabel(r"$f$ $\mathrm{[Hz]}$", labelpad=-5)

        # Make more space for colorbar
        fig.subplots_adjust(right=0.85, hspace=0.4, wspace=0.5)

        # Finally add labels
        # labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
        # axs = [ax_upper, ax_11, ax_12, ax_21, ax_22]
        #
        # for ax, lab in zip(axs, labels):
        #     ax.text(
        #         0.05, 0.95, lab,
        #         transform=ax.transAxes,
        #         va="top",
        #         ha="left",
        #         fontsize=label_fontsize_pts,
        #     )

        y1 = 0.495
        plt.annotate(
            "",
            xytext=(0.43, y1),
            xy=(0.47, y1),
            xycoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="k", lw=3)
        )

        y2 = 0.212
        plt.annotate(
            "",
            xytext=(0.43, y2),
            xy=(0.47, y2),
            xycoords='figure fraction',
            arrowprops=dict(arrowstyle="->", color="k", lw=3)
        )

        save_figure(save_fig=True, tight_ly=False, show=True)

