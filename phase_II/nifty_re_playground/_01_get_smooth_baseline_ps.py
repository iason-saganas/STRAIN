from phase_II.nifty_re_playground.useful.helpers import *
jax.config.update("jax_enable_x64", True)
key = jax.random.key(42)

if __name__ == "__main__":
    pipe_1_called_as_import = False
else:
    pipe_1_called_as_import = True

time, strain = get_sample_data(end_points_small=False, taper=False)

pipe_1=InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key, plotting_callback=analyze_kl_callback)
pipe_1.add_cfm_signal_model(fluct=(4, 2), llslope=(-2, 1), flex=(.5, .1), square_iwp=True)
pipe_1.add_noise_op(noise_var_level=.1)

if not pipe_1_called_as_import:
    pipe_1.plot_prior_samples(mode="power spectrum", num=5, plot_welch_average=True)

    pipe_1.plot_prior_samples(mode="signal", num=5)
    for _ in range(0):
        sl = pipe_1.plot_prior_samples(mode="signal", num=1, plot=False)
        plt.plot(pipe_1.t_ss, sl, label="Sample")
        plt.plot(pipe_1.t_ds, strain, label="Actual data")
        usual_plot(xl="Time", yl="Strain", title="Data with signal sample (extended domain for periodicity)")

latent_samples = pipe_1.run_inference(kl_iterations=30, use_strict_minimizers=False, out_name="small_data_set/re_pipe_1", resume=True,
                                      choose_low_kl_starting_pos=not pipe_1_called_as_import, geoVi=True)
key = pipe_1.get_current_key()

if not pipe_1_called_as_import:
    pipe_1.plot_posterior_signal(save_fig=True)
    pipe_1.plot_posterior_power_spectrum(mode="mean", print_posterior_parameters=True, save_fig=True)
    pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=False)
    pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)