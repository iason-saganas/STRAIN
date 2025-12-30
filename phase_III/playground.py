from autodiff_response_solver import *

file = "sde3"
all_latent_means = [unpickle_me_this(f) for f in os.listdir(f"{file}/latent_means/")]

generative_wavelet_adaptive_solver = AutoDiffEquationSolver(
    prefix="stochastic_diff_equ_",
    reconstruction_times=gw_times,
    cfm_sampling_times=cfm_times,
    omega_cfm=omega_cfm,
    gamma_cfm=gamma_cfm,
    xi_cfm=xi_cfm,
    scaling_constant=scaling_constant,
    solver=diffrax_solver
)

for latent_mean in all_latent_means:
    sample_waveform, key = draw_and_plot_field_realizations(times=cfm_times, diff_eq_solver_model=generative_wavelet,
                                                            omega_op=omega_cfm, gamma_op=gamma_cfm, xi_op=xi_cfm,
                                                            key=key, custom_latent_position=latent_mean)

    sample_waveform, key = draw_and_plot_field_realizations(times=cfm_times, diff_eq_solver_model=generative_wavelet_adaptive_solver,
                                                            omega_op=omega_cfm, gamma_op=gamma_cfm, xi_op=xi_cfm,
                                                            key=key, custom_latent_position=latent_mean)
