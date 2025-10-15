from psfhs_pipe_1 import *
from scipy.interpolate import interp1d

# Get the gaussian comb operator. Initialized by a human!
assert np.allclose(np.diff(f), np.diff(f)[0], rtol=0, atol=1e-10)
df = np.diff(f)[0]
# (index of frequency at peak, amplitude of peak, half-width of the peak in INDEX counts)
information_idx = list(np.loadtxt("information_idcs_2_3.txt"))
information_idx = [tuple(el) for el in information_idx]
information_idx = [(np.int64(tl[0]), tl[1], tl[2]) for tl in information_idx]

# => Convert idx_center and idx_width to freq_center and freq_width
information_freq = [( f[t[0]],  t[1], t[2] * df) for t in information_idx]
frequency_centers = [t[0] for t in information_freq]
peak_amplitudes = [t[1] for t in information_freq]
frequency_half_widths = [t[2] for t in information_freq]

h_dom = inference_scheme_pipe_1.domain_ext.get_default_codomain()
p_space = ift.PowerSpace(h_dom)
frequency_field = ift.Field(ift.DomainTuple.make(p_space, ), val=f)

const = 1e3
lorentzian_comb_op = const * generative_gaussian_comb(x_field=frequency_field, position_of_peaks=frequency_centers,
                                            amplitude_of_peaks=peak_amplitudes, half_width_of_peaks=frequency_half_widths,
                                                    rel_sigma_lognormal=.5, vary_amplitudes=True, vary_positions=True,
                                                    rel_sigma_normal=1)


# Start with a fresh whiten noise harmonic xi_s
inference_scheme_pipe_2 = ExecuteEasyRGSpaceKL(
    discrete_time=time,
    d=data,
    cfm_model_name="s_",
    gaussian_noise_level=1e-10,
    out_dir_name="outs/power_spectrum_from_harmonic_stress_f_pipe_2/",
    fluct=(const, const*10),  # good-looking samples: (const, 1e-16) where const = 70 | (const, const*.1)
    llslope=(1, 1e-16),  # good-looking samples: (1, 1e-16) + infer peak amplitude from a new i.e. amp_mean, amp_sig = (0,1) | (post_pipe_1_slope_mean, 2)
    # flex=(1, 1),
    # asper=(5, 1),
    custom_generative_model=None,
    kl_minimizations=19,
    op_to_apply_to_amp=(lorentzian_comb_op, "multiply", True),
    x_fac=2,
    n_pix_fac=1,
    mgvi_controler=ic_sampling_lin,
    geo_vi_minimizer=geoVI_sampling_minimizer,
    kl_minimizer=descent_finder
)

ns = 1
# ns = 15
for _ in range(ns):
    inference_scheme_pipe_2.plot_amplitude_spectrum_prior_samples(1, "-", plot_welch_average=True)

stop

for _ in range(ns):
    plt.plot(time, data, "g.")
    usual_plot()
    # inference_scheme_pipe_2.plot_prior_samples(num=1, ls=".")

inference_scheme_pipe_2.run()

inference_scheme_pipe_2.plot_posterior(plot_signal_space=True)
inference_scheme_pipe_2.plot_posterior(plot_signal_space=False)

k_domain_lengths, mean_amp_spec = inference_scheme_pipe_2.plot_posterior_amp_spec(show=True, plot_welch_average=True)


p_sls = inference_scheme_pipe_2.posterior_samples
post_mean_latent, post_var_latent = p_sls.sample_stat()
mdl = inference_scheme_pipe_2.model

# ---- Execute whiten and bandpass quality check of posterior reconstruction

config_for_quality_check = {
    "space": 'signal_space',  # 'signal_space' | 'data_space'
    "data_to_use": 'model_prediction' # 'model_prediction' | 'actual_data'
}

space = config_for_quality_check.get("space")
data_to_use = config_for_quality_check.get("data_to_use")

if space == "data_space":

    # get coarse interpolation of asd...
    my_interp = interp1d(k_domain_lengths, mean_amp_spec.val, kind="linear", fill_value="extrapolate",
                         assume_sorted=False)
    h_data_space = inference_scheme_pipe_2.data_space.get_default_codomain()
    coarser_data_asd = my_interp(h_data_space.get_unique_k_lengths())
    p_data_space = ift.PowerSpace(h_data_space)
    coarser_data_asd_field = fieldify(coarser_data_asd, p_data_space)

    asd_fld = coarser_data_asd_field
    x_fld = inference_scheme_pipe_2.discrete_domain_values

    if data_to_use == "actual_data":
        y_fld = inference_scheme_pipe_2.data_field
    elif data_to_use == "model_prediction":
        y_fld = inference_scheme_pipe_2._R_full(post_mean_latent)
    else:
        raise ValueError(f"Unknown data_to_use: {data_to_use}")

elif space == "signal_space":
    asd_fld = mean_amp_spec
    x_fld = inference_scheme_pipe_2.domain_values_ext.val
    signal_mean = mdl(post_mean_latent)
    y_fld = signal_mean

else:
    raise ValueError(f"Unknown space: {space}")

nt = space if space == "signal_space" else space + " " + data_to_use
check_quality_of_psd_by_whitening(
    x=x_fld,
    y_field=y_fld,
    asd_on_power_space=asd_fld,
    plot_wh_and_bp=True,
    plot_stress_of_wh_data=False,
    cut_x=np.max(inference_scheme_pipe_2.discrete_domain_values),
    notes=nt
)

try:
    post_s_xi_std = np.sqrt(post_var_latent["s_xi"])  # length == length of extended signal domain
    post_s_xi_mean = post_mean_latent["s_xi"]  # length == length of extended signal domain
    xi_subdomain = mdl.domain["s_xi"]
except KeyError:
    post_s_xi_std = np.sqrt(post_var_latent["xi_s"])  # length == length of extended signal domain
    post_s_xi_mean = post_mean_latent["xi_s"]  # For the broken power law model
    xi_subdomain = mdl.domain["xi_s"]


### ---- Get zero-centered posterior harmonic xi and frequencies for plotting

post_s_xi = hartley_to_fftshift(post_s_xi_mean.val)
freqs = hartley_to_fftshift(xi_subdomain[0].get_k_length_array().val, flip_negatives=True)


### ---- Plot posterior harmonix xi_s
# plt.errorbar(freqs, post_s_xi, yerr=post_s_xi_std.val, color="green", ecolor="lightgreen", label=r"Posterior harmonic $\xi_s$", zorder=1)
plt.plot(freqs, post_s_xi, "r-", label=r"Posterior harmonic $\xi_s$", zorder=2)
usual_plot(xl="Frequency $f$", yl=r"$\xi_s$")

plot_wigner=False
if plot_wigner:

    ### ---- Calculation of Wigner function

    wigner_exists = False
    try:
        np.loadtxt("i don't exist.txt")
        wigner_result = unpickle_me_this("wigner_result_pipe_2.pickle")

    except FileNotFoundError:

        fft_helper = ift.FFTOperator(space=0, domain=xi_subdomain)

        real_xi =  fft_helper(post_s_xi_mean)

        cutter = inference_scheme_pipe_1.X
        masker = inference_scheme_pipe_1.R_physical

        xi_real_cut_and_masked = masker(cutter.adjoint(real_xi))

        wigner_result  = Stress(xi_real_cut_and_masked)
        # pickle_me_this("wigner_result_pipe_2", wigner_result)


    wigner_mat, t_dual, freqs = wigner_result

    print("\tMean and std of wigner: ", np.mean(wigner_mat.real), np.std(wigner_mat.real))

    ## ---- Print loud frequencies ( = rows/y-axis in imatshow plot)

    wr = wigner_mat.real
    # loud_threshhold = 1e7
    # loud_idcs = np.where(wr > loud_threshhold)[0]
    # loud_freqs = np.unique(freqs[loud_idcs])
    #
    # print(f"Loud (stress > {loud_threshhold}) frequencies are : ",loud_freqs)


    ### ---- Set the DC-strip to 0 which contains very strong auto-terms and positively interfering cross-terms (even larger)

    to_cut = ~((30 < freqs) & (freqs < 800)) # cut out the loud interference part at the 0 frequency...
    wigner_mat = wr.copy()
    wigner_mat[to_cut,:] = 0

    ### ---- Finally visualize the wigner function and a frequency-marginalized version

    visualize_stress(wigner_mat, rows=freqs, cols=t_dual+time[0])

    marginalized = np.sum(wigner_mat, axis=0)
    plt.plot(t_dual+time[0], marginalized)
    usual_plot(yl="Stress", title="Frequency-Marginalized stress evolution")