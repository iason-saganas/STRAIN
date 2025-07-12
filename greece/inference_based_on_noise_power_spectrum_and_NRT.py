from quickview.greece.utils.generative_models import *

inference_scheme = ExecuteRGSpaceKL(
    custom_data_space = N.domain[0],
    discrete_time=signal_strip_time,
    d=signal_strip_strain_tapered,
    cfm_model_name="GW numerical wavelet model ",
    kl_minimizations=15,
    fluct=(8, 1),
    llslope=(-4, 0.5),
    gaussian_noise_level=None,
    custom_noise_operator=N,
    out_dir_name="outs/numerical_wavelet_spline_model_out_6 ",
    custom_generative_model=wavelet_model_spline
)

inference_scheme.run()

inference_scheme.get_posterior_parameters()

plt.plot(nrt_time_values, nrt_strain_values, label="suggested numerical relativity template", color="green")
inference_scheme.plot_posterior(plot_with_variance=True, plot_signal_space=True)


