from strain_tools import *
import matplotlib.pyplot as plt
import jax

jax.config.update("jax_enable_x64", True)
key = jax.random.key(42)

time, strain = get_sample_data(end_points_small=False, taper=False)
data_middle_line, std_data = iterative_midpoint_average(strain, plot=False, n_iter=10)

additional_noise_var_lvl = (3*std_data) ** 2  # we want to ignore small scale structure, so add 3 sigma as uncertainty
chi_sq_ad_hoc_threshold = 0.2

if __name__ == "__main__":
    pipe_1_called_as_import = False
else:
    pipe_1_called_as_import = True

pipe_1 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key, plotting_callback=analyze_kl_callback)
pipe_1.add_cfm_signal_model(fluct=(4, 2), llslope=(-2, 1), flex=(.5, .1), square_iwp=False, apply_tukey_window=False)

# N_cov = jft.LogNormalPrior(mean=1, std=1e-32, name="Additional noise level", dtype=jnp.float64, shape=())
# N_inv_cov = jft.Model(domain=N_cov.domain, call=lambda xi: N_cov(xi)**(-1))
# N_inv_std_cov = jft.Model(domain=N_cov.domain, call=lambda xi: jnp.sqrt(N_inv_cov(xi)))
# pipe_1.add_noise_op(inverse_noise_op=N_inv_cov, sqrt_inverse_noise_op=N_inv_std_cov)  # to ignore small scales: σ_n ~ .2 => Var(n) = 0.0004 => nan energy error in geoVi and power spectrum underflows

pipe_1.add_noise_op(noise_var_level=additional_noise_var_lvl)  # to ignore small scales: σ_n ~ .2 => Var(n) = 0.0004 => nan energy error in geoVi and power spectrum underflows

plot_prior_samples = True
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
                                               out_name="re_pipe_1", resume=True, geoVi=True,
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