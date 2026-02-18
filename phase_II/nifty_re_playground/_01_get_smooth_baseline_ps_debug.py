import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

from strain_tools import *
# In the future: start the baseline guess at the smooth baseline of the Welch average

jax.config.update("jax_enable_x64", True)
key = jax.random.key(42)

time, strain = get_sample_data(end_points_small=False, taper=False)

data_middle_line, std_data = iterative_midpoint_average(strain, plot=False, n_iter=10)


strain = strain - jnp.mean(strain)

additional_noise_var_lvl = (3*std_data) ** 2  # we want to ignore small scale structure, so add 3 sigma as uncertainty
chi_sq_ad_hoc_threshold = 0.2
print("Assumed fiducial noise variance level: ", additional_noise_var_lvl)
print("Ad-hoc chosen red_chi2 treshhold:", chi_sq_ad_hoc_threshold)


if __name__ == "__main__":
    pipe_1_called_as_import = False
else:
    pipe_1_called_as_import = True


pipe_1 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key, plotting_callback=analyze_kl_callback)
pipe_1.add_cfm_signal_model(fluct=(np.std(strain), 1), llslope=(-2, 1), flex=(2, 2), square_iwp=False)

# N_cov = jft.LogNormalPrior(mean=1, std=1e-32, name="Additional noise level", dtype=jnp.float64, shape=())
# N_inv_cov = jft.Model(domain=N_cov.domain, call=lambda xi: N_cov(xi)**(-1))
# N_inv_std_cov = jft.Model(domain=N_cov.domain, call=lambda xi: jnp.sqrt(N_inv_cov(xi)))
# pipe_1.add_noise_op(inverse_noise_op=N_inv_cov, sqrt_inverse_noise_op=N_inv_std_cov)  # to ignore small scales: σ_n ~ .2 => Var(n) = 0.0004 => nan energy error in geoVi and power spectrum underflows

pipe_1.add_noise_op(noise_var_level=additional_noise_var_lvl)  # to ignore small scales: σ_n ~ .2 => Var(n) = 0.0004 => nan energy error in geoVi and power spectrum underflows

plot_prior_samples = False  # set this to True for good inference seed
if not pipe_1_called_as_import and plot_prior_samples:
    pipe_1.plot_prior_samples(mode="power spectrum", num=5, plot_welch_average=True, plot_data=False)


    pipe_1.plot_prior_samples(mode="signal", num=1)
    for _ in range(1):
        sl = pipe_1.plot_prior_samples(mode="signal", num=1, plot=False)
        plt.plot(pipe_1.t_ss, sl, label="Sample")
        plt.plot(pipe_1.t_ds, strain, label="Actual data")
        usual_plot(xl="Time", yl="Strain", title="Data with signal sample (extended domain for periodicity)")

pipe_1_base_iterations = 10  # minimum number of iterations to assure convergence
latent_samples, vi_info = pipe_1.run_inference(kl_iterations=pipe_1_base_iterations, use_strict_minimizers=False,
                                               out_name="re_pipe_1_debug", resume=True, geoVi=True,
                                               choose_low_kl_starting_pos=False, max_kl_iter=20,
                                               chi2_threshold=chi_sq_ad_hoc_threshold)


key = pipe_1.get_current_key()

if not pipe_1_called_as_import:
    pipe_1.plot_posterior_signal(save_fig=False)
    pipe_1.plot_posterior_power_spectrum(mode="mean", plot_welch_average=True, save_fig=False)
    # pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=False)
    #
    # pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)
    # pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec_v2=True)

    #### THESIS PLOT
    thesis_plot_1 = True

    if thesis_plot_1:
        fig, axs = plt.subplots(2, 1, figsize=(8, 4*2), sharex=True, sharey=True)

        post_xi_samples = pipe_1.posterior_xi_samples
        ps = lambda x: pipe_1.amplitude_op(x) ** 2
        ps_samples = jnp.array([ps(xi) for xi in post_xi_samples])
        ps_mean = jnp.mean(ps_samples, axis=0)

        pipe_1.plot_prior_samples(mode="power spectrum", num=7, custom_ax=axs[0], show=False, plot_data=False)
        plot_welch_averaged_ps(ax=axs[0])

        plot_welch_averaged_ps(ax=axs[1], lb="")
        axs[1].plot(pipe_1.k_signal[1:], ps_mean[1:], label="Posterior mean", color=blue, lw=2)

        axs[1].set_xlabel(r"Frequency $f$ $\mathrm{[Hz]}$")
        axs[0].set_ylabel(r"Power")
        axs[1].set_ylabel(r"Power")
        axs[0].legend()
        axs[1].legend()
        axs[0].loglog()
        axs[1].loglog()
        save_figure(save_fig=False, show=True, tight_ly=True)