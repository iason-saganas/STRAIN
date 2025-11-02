from helpers import *
jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(49)

"""
Good parameters: 

    CFM with fluct=(4, 2), llslope=(-2, 1), flex=(2, 1)
    noise_sigma_squared = 1e-10
    use_strict_minimizers=True
    
Alternatively:

    CFM with fluct=(4, 2), llslope=(-2, 1), flex=(2, 1)
    noise_sigma_squared = .1
    use_strict_minimizers=False

"""

if __name__ == "__main__":
    pipe_1_called_as_import = False
else:
    pipe_1_called_as_import = True

time, strain = get_sample_data()

pipe_1 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key)

# custom_signal_model = SignalModelCfmAsPowerSpectrum(scale=(1e-8, 1e-16), llslope=(2, 1e-16),  #asper = (1,1), flex=(1, 1),
#                                                     N_ss=pipe_1.n_ss, dist_ss=pipe_1.dist_ss, offset_mean=-1)

# pipe_1.add_custom_signal_model(custom_signal_model)
pipe_1.add_cfm_signal_model(fluct=(4, 2), llslope=(-2, 1), flex=(2, 1),) #asper=(2,1))

pipe_1.add_noise_op(noise_var_level=1e-10)

if not pipe_1_called_as_import:
    pipe_1.plot_prior_samples(mode="power spectrum", num=5, plot_welch_average=True)
    # pipe_1.plot_prior_samples(mode="signal", num=5, plot_welch_average=True)

    num_signal_samples = 1

    for _ in range(5):
        sl = pipe_1.plot_prior_samples(mode="signal", num=1, plot=False)
        plt.plot(pipe_1.t_ss, sl, label="Sample")
        plt.plot(pipe_1.t_ds, strain, label="Actual data")
        usual_plot(xl="Time", yl="Strain", title="Data with signal sample (extended domain for periodicity)")

latent_samples = pipe_1.run_inference(kl_iterations=10, use_strict_minimizers=True, out_name="re_pipe_1", resume=True)
key = pipe_1.get_current_key()

if not pipe_1_called_as_import:
    pipe_1.plot_posterior_signal()
    pipe_1.plot_posterior_power_spectrum(print_posterior_parameters=True)
    pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=False)
    # pipe_1.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)