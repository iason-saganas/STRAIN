from re_psfhs_pipe_1 import *

if __name__ == "__main__":
    pipe_2_called_as_import = False
else:
    pipe_2_called_as_import = True

time, strain = get_sample_data()

pipe_2 = InferenceSchemeRe(t=time, d=strain, e_fac=2, r_fac=1, key=key)

prior_ps_template_mean_std, _, _ = pipe_1.get_posterior_statistics()
prior_ps_template_mean = prior_ps_template_mean_std[0]

pipe_2.add_cfm_signal_model(fluct=(4, 2), llslope=(-2, 1), flex=(2, 1), add_power_spectrum_template=prior_ps_template_mean)

pipe_2.add_noise_op(noise_var_level=1e-10)

if not pipe_2_called_as_import:

    plt.plot(pipe_1.k_signal, prior_ps_template_mean, label="Prior template")
    pipe_2.plot_prior_samples(mode="power spectrum", num=5, plot_welch_average=True)

    for _ in range(5):
        sl = pipe_2.plot_prior_samples(mode="signal", num=1, plot=False)
        plt.plot(sl, label="Sample")
        plt.plot(strain, label="Actual data")
        plt.legend()
        plt.show()


latent_samples = pipe_2.run_inference(kl_iterations=10, use_strict_minimizers=True, out_name="re_pipe_2", resume=True)
key = pipe_2.get_current_key()

if not pipe_2_called_as_import:

    pipe_2.plot_posterior_signal()
    pipe_2.plot_posterior_power_spectrum(print_posterior_parameters=True)
    pipe_2.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=False)
    pipe_2.plot_posterior_harmonic_xi_s(multiply_with_posterior_amp_spec=True)