from psfhs_pipe_1 import *
from scipy.interpolate import interp1d

# Get the gaussian comb operator. Initialized by a human!
assert np.allclose(np.diff(f), np.diff(f)[0], rtol=0, atol=1e-10)
df = np.diff(f)[0]
# (index of frequency at peak, amplitude of peak, half-width of the peak in INDEX counts)
information_idx = list(np.loadtxt("information_idcs.txt"))
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

feed_into_third_pipe = False

gaussian_comb_op =  generative_gaussian_comb(x_field=frequency_field, position_of_peaks=frequency_centers,
                                                    amplitude_of_peaks=peak_amplitudes, half_width_of_peaks=frequency_half_widths,
                                                    vary_amplitudes=True, vary_positions=False, rel_sigma_lognormal=.5
                                             )

template_from_pipe_1 = generative_asd_template_model(k_values=k_domain_lengths, asd_template_values=envelope_vals,
                                                     extended_real_space_domain=inference_scheme_pipe_1.domain_ext,
                                                     additional_operator_to_add=gaussian_comb_op, amp=(1e4,1e4),
                                                     )

# Start with a fresh whiten noise harmonic xi_s
inference_scheme_pipe_2 = ExecuteEasyRGSpaceKL(
    discrete_time=time,
    d=data,
    cfm_model_name="s_",
    gaussian_noise_level=1e-10,
    out_dir_name="outs/power_spectrum_from_harmonic_stress_f_pipe_2_new",
    fluct=(1, 1e-16),
    llslope=(0, 1e-16),
    flex=(.1, .1),
    custom_generative_model=None,
    kl_minimizations=10,
    op_to_apply_to_amp=(template_from_pipe_1, "multiply", False),
    x_fac=2,
    n_pix_fac=1,
    mgvi_controler=ic_sampling_lin,
    geo_vi_minimizer=geoVI_sampling_minimizer,
    kl_minimizer=descent_finder
)

# k_tmp, prior_amp_spec_samples = inference_scheme_pipe_2.plot_amplitude_spectrum_prior_samples(50, "-", plot_welch_average=False, show=False)
# stop
ns = 3
# ns = 15
for _ in range(ns):
    # _ = plt.figure(figsize=(10, 6))
    # plt.errorbar(k_tmp, np.mean(prior_amp_spec_samples, axis=0), label="Mean prior amplitude spectrum",
    #              yerr=np.std(prior_amp_spec_samples, axis=0), ecolor="lightblue")
    inference_scheme_pipe_2.plot_amplitude_spectrum_prior_samples(1, "-", plot_welch_average=True, exists_figure=False)


for _ in range(ns):
    plt.plot(time, data, "g.")
    inference_scheme_pipe_2.plot_prior_samples(num=1, ls=".")

inference_scheme_pipe_2.run()

inference_scheme_pipe_2.plot_posterior(plot_signal_space=True)
inference_scheme_pipe_2.plot_posterior(plot_signal_space=False)

k_domain_lengths, mean_amp_spec = inference_scheme_pipe_2.plot_posterior_amp_spec(show=True, plot_welch_average=True)
np.savetxt("pipe_2_post_amp_spec.txt", mean_amp_spec.val)

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


post_s_xi_mean_2 = post_s_xi_mean.val.copy()

post_s_xi_mean_2[np.where(post_s_xi_mean_2 > 3.29)] = 0
post_s_xi_mean_2[np.where(post_s_xi_mean_2 < -3.29)] = 0
post_s_xi_mean = ift.makeField(domain=post_s_xi_mean.domain, arr=post_s_xi_mean_2)

### --- Plotting power spectrum * xi

PD = ift.PowerDistributor(target=mean_amp_spec.domain[0]._harmonic_partner)
full_amp_xi_spec = PD(mean_amp_spec) * np.sqrt(post_s_xi_mean**2)
amp_xi_spec = PD.adjoint(full_amp_xi_spec)

envelope_vals = extract_envelope(k_domain_lengths, amp_xi_spec.val,
                                 win=31, perc=95, sg_window=201, sg_poly=3)

if not feed_into_third_pipe:

    _, k_lengths, power_spectrum = unpickle_me_this(
                    "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                    absolute_path=True)
    k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
    amp_spectrum_welch = np.sqrt(power_spectrum.val[1:])

    # np.savetxt("pipe_2_xi_s_amp_spec_envelope_y.txt", envelope_vals)
    # np.savetxt("pipe_2_xi_s_amp_spec_envelope_x.txt", k_domain_lengths)

    plt.plot(k_domain_lengths, amp_xi_spec.val,
             label=r"Post amp spec $\cdot \mid \xi_s \mid$ posterior")
    plt.plot(k_domain_lengths, mean_amp_spec.val,
             label=r"Post amp spec")
    plt.plot(k_lengths, amp_spectrum_welch, label="Empirical estimate")

    # Plot extracted envelope
    plt.plot(k_domain_lengths, envelope_vals, 'k--', lw=2, label=r"Updated amplitude spectrum (containg $\xi_s$ contr.)")

    plt.loglog()
    plt.legend()
    plt.show()

### ---- Get zero-centered posterior harmonic xi and frequencies for plotting

post_s_xi = hartley_to_fftshift(post_s_xi_mean.val)
freqs = hartley_to_fftshift(xi_subdomain[0].get_k_length_array().val, flip_negatives=True)


### ---- Manual peak detection using simple threshhold from zero-centered xi_s

threshhold_plus = 1

use_prominence = False
use_sigma_as_threshhold = False
if use_sigma_as_threshhold:

    f_minus = 540  # lower bound of frequency
    f_plus = 700  # upper bound ''. The statistics of xi are estimated inbetween these two points.

    x_estimate = np.where((freqs>f_minus) & (freqs<f_plus))
    y_estimate = post_s_xi[x_estimate]
    sig = np.std(y_estimate)

    fac = 3
    threshhold = fac * sig

    print(f"Using posterior xi_s in between frequencies ({f_minus},{f_plus}) to determine threshhold as {fac} * {sig} = "
          f"{fac * sig}")

y_hits = np.zeros(len(freqs))

if use_prominence:
    idx_hits, props = find_peaks(post_s_xi, prominence=.2)

else:

    threshhold_minus = -np.inf
    idx_hits_plus = np.where(post_s_xi >= threshhold_plus)
    idx_hits_minus = np.where(post_s_xi <= threshhold_minus)

    idx_hits = np.append(idx_hits_plus, idx_hits_minus).flatten()

y_hits[idx_hits] = post_s_xi[idx_hits]

freqs_dtf_order = fftshift_to_hartley(freqs)
y_hits_dtf_order = fftshift_to_hartley(y_hits)

if np.abs(np.max(freqs_dtf_order)) < np.abs(np.min(freqs_dtf_order)):
    raise ValueError("The nyquist frequency is negative, although it should be positive in the correct ordering.")
# else:
  # The nyquist frequency is indeed the maximum of the array => Use it to find periodic boundary

location_positive_nyquist = np.where(freqs_dtf_order == np.max(freqs_dtf_order))[0][0]
if feed_into_second_pipe:
    f = freqs_dtf_order[:location_positive_nyquist+1]
else:
    np.savetxt("spike_frequencies_from_pipe_2.txt", freqs_dtf_order[:location_positive_nyquist+1])
    np.savetxt("spike_amplitudes_from_pipe_2.txt", y_hits_dtf_order[:location_positive_nyquist+1])
    print("\n Saved found peak position and amplitudes.")

    # Both of the saved arrays are in standard DFT order, i.e. the DC frequency k=0 first.
    # Because we fundamentally want to apply this to the amplitude operator, which is defined on the power domain
    # instead of the harmonic domain, we only want the positive frequency part of the peaks, without any periodic
    # repetitions. We add +1 to the slicing to include the nyquist frequency.

    pass


### ---- Manually subtract estimated peaks for investigation purposes

# fld = fieldify(array=y_hits, dom=xi_subdomain)
# post_s_xi_mean = post_s_xi_mean - fld


### ---- Plot posterior harmonix xi_s
if not feed_into_third_pipe:
    # plt.errorbar(freqs, post_s_xi_mean, yerr=post_s_xi_std, color="green", ecolor="lightgreen", label=r"Posterior harmonic $\xi_s$", zorder=1)
    plt.plot(freqs, post_s_xi, "r-", label=r"Posterior harmonic $\xi_s$", zorder=2)
    usual_plot(xl="Frequency $f$", yl=r"$\xi_s$")


plot_wigner=True
if plot_wigner:

    ### ---- Calculation of Wigner function

    wigner_exists = False
    try:
        np.loadtxt("I don't exist.txt")
        wigner_result = unpickle_me_this("wigner_result_pipe_1.pickle")

    except FileNotFoundError:

        fft_helper = ift.FFTOperator(space=0, domain=xi_subdomain)

        real_xi =  fft_helper(post_s_xi_mean)

        cutter = inference_scheme_pipe_1.X
        masker = inference_scheme_pipe_1.R_physical

        xi_real_cut_and_masked = masker(cutter.adjoint(real_xi))

        wigner_result  = Stress(xi_real_cut_and_masked)
        pickle_me_this("wigner_result_pipe_2", wigner_result)


    wigner_mat, t_dual, freqs = wigner_result


    ## ---- Print loud frequencies ( = rows/y-axis in imatshow plot)

    wr = wigner_mat.real
    loud_threshhold = 1e7
    loud_idcs = np.where(wr > loud_threshhold)[0]
    loud_freqs = np.unique(freqs[loud_idcs])

    print(f"Loud (stress > {loud_threshhold}) frequencies are : ",loud_freqs)


    ### ---- Set the DC-strip to 0 which contains very strong auto-terms and positively interfering cross-terms (even larger)

    # to_cut = (-50 < freqs) & (freqs < 50)  # cut out the loud interference part at the 0 frequency...
    # wigner_mat = wr.copy()
    # wigner_mat[to_cut,:] = 0

    ### ---- Finally visualize the wigner function and a frequency-marginalized version

    visualize_stress(wigner_mat, rows=freqs, cols=t_dual+time[0])

    # marginalized = np.sum(wigner_mat, axis=0)
    # plt.plot(t_dual+time[0], marginalized)
    # usual_plot(yl="Stress", title="Frequency-Marginalized stress evolution")

if feed_into_third_pipe:

    post_params = inference_scheme_pipe_1.get_posterior_parameters()
    post_fluct = post_params["s_fluctuations"]
    post_llslope = post_params["s_loglogavgslope"]

    try:
        post_flex = post_params["s_flexibility"]
        post_pipe_1_flex_mean = post_flex[0]
        post_pipe_1_flex_std = post_flex[1]
    except KeyError:
        pass

    post_pipe_1_slope_mean = post_llslope[0]
    post_pipe_1_slope_std = post_llslope[1]

    post_pipe_1_fluct_mean = post_fluct[0]
    post_pipe_1_fluct_std = post_fluct[1]


    print("\n\n\n\nEntering third pipe...\n")