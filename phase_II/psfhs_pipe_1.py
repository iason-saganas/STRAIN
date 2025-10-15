from scipy.signal import find_peaks
from phase_I.utils.config_jupyter_notebooks import *
from utils.helpers import *

# The abbreviation stands for: power spectrum from harmonic stress, pipe 1

full_data = 1e19 * strain.value
full_time = time.copy()

indcs = np.where((15<time) & (time<17))
data = data[indcs]
time = time[indcs]

# data = data * window_function  # do NOT taper; that's what the Zeropadder is for!!

ift.random.push_sseq_from_seed(27)

# data = np.loadtxt("tmp_rnd_data.txt")

def model_wrap(real_dom_ext, real_dom_ext_values):
    h = real_dom_ext.get_default_codomain()
    s_broken_power_law = generative_model_continuous_double_power_law(h, apply_envelope=False)
    return s_broken_power_law

# plt.plot(time, data)
# usual_plot()

feed_into_second_pipe = True

# --- More accurate minimizers...

use_strict_minimizers = True

if use_strict_minimizers:
    # Iteration control for `MGVI` and linear parts of the inference
    ic_sampling_lin = ift.AbsDeltaEnergyController(name="Precise linear sampling", deltaE=0.02, iteration_limit=100)

    # Iteration control for `geoVI`
    ic_sampling_nl = ift.AbsDeltaEnergyController(name="Coarser, nonlinear sampling", deltaE=0.5, iteration_limit=20,
                                                  convergence_level=2)  # geoVI is from standard minimization control file
    # since the problem is per-se linear, we don't need to rely on geoVi as much I assume...
    # For the non-linear sampling part of geoVI, the iteration controller has to be "promoted" to a minimizer:

    geoVI_sampling_minimizer = ift.NewtonCG(ic_sampling_nl)

    # KL Minimizer control, the same energy criterion as the geoVI iteration control, but more iteration steps
    ic_newton = ift.AbsDeltaEnergyController(name='Newton Descent Finder', deltaE=0.1, convergence_level=2,
                                             iteration_limit=50)

    descent_finder = ift.NewtonCG(ic_newton)

inference_scheme_pipe_1 = ExecuteEasyRGSpaceKL(
    discrete_time=time,
    d=data,
    cfm_model_name="s_",
    gaussian_noise_level=1e-10,
    out_dir_name="outs/power_spectrum_from_harmonic_stress_f_pipe_1",
    fluct=(5,2),
    llslope=(-4, 2),
    custom_generative_model=None,
    kl_minimizations=22,
    # flex=(1, 1),
    x_fac=2,
    n_pix_fac=1,
    mgvi_controler=ic_sampling_lin,
    geo_vi_minimizer=geoVI_sampling_minimizer,
    kl_minimizer=descent_finder
)


# response = inference_scheme._R_full
# rnd_data = response(ift.from_random(response.domain))
# np.savetxt("tmp_rnd_data.txt", rnd_data.val)
#
if not feed_into_second_pipe:
    inference_scheme_pipe_1.plot_amplitude_spectrum_prior_samples(num=3, ls="-")
    plt.show()

    plt.plot(time, data)
    inference_scheme_pipe_1.plot_prior_samples(num=2)
    plt.show()

inference_scheme_pipe_1.run()

if not feed_into_second_pipe:
    inference_scheme_pipe_1.plot_posterior(plot_signal_space=False)
    inference_scheme_pipe_1.plot_posterior(plot_signal_space=True)

k_domain_lengths, mean_amp_spec = inference_scheme_pipe_1.plot_posterior_amp_spec(show=not feed_into_second_pipe, plot_welch_average=True)

np.savetxt("some_posterior_amp_spec.txt", mean_amp_spec.val)

p_sls = inference_scheme_pipe_1.posterior_samples
post_mean, post_var = p_sls.sample_stat()

mdl = inference_scheme_pipe_1.model

try:
    post_s_xi_std = np.sqrt(post_var["s_xi"])  # length == length of extended signal domain
    post_s_xi_mean = post_mean["s_xi"]  # length == length of extended signal domain
    xi_subdomain = mdl.domain["s_xi"]
except KeyError:
    post_s_xi_std = np.sqrt(post_var["xi_s"])  # length == length of extended signal domain
    post_s_xi_mean = post_mean["xi_s"]  # For the broken power law model
    xi_subdomain = mdl.domain["xi_s"]

### --- Plotting power spectrum * xi

PD = ift.PowerDistributor(target=mean_amp_spec.domain[0]._harmonic_partner)
full_amp_xi_spec = PD(mean_amp_spec) * np.sqrt(post_s_xi_mean**2)
amp_xi_spec = PD.adjoint(full_amp_xi_spec)

envelope_vals = extract_envelope(k_domain_lengths, amp_xi_spec.val,
                                 win=31, perc=95, sg_window=201, sg_poly=3)

if not feed_into_second_pipe:

    _, k_lengths, power_spectrum = unpickle_me_this(
                    "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                    absolute_path=True)
    k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
    amp_spectrum_welch = np.sqrt(power_spectrum.val[1:])

    np.savetxt("pipe_1_xi_s_amp_spec_envelope_y.txt", envelope_vals)
    np.savetxt("pipe_1_xi_s_amp_spec_envelope_x.txt", k_domain_lengths)

    plt.plot(k_domain_lengths, amp_xi_spec.val, label=r"Post amp spec $\cdot \mid \xi_s \mid$ posterior")
    plt.plot(k_domain_lengths, mean_amp_spec.val, label=r"Post amp spec")
    plt.plot(k_lengths, amp_spectrum_welch, label="Empirical estimate")

    # Plot extracted envelope
    plt.plot(k_domain_lengths, envelope_vals, 'k--', lw=2, label=r"Updated amplitude spectrum (containg $\xi_s$ contr.)")

    plt.xlabel(r"Unique $\omega$")
    plt.ylabel(r"Amplitude spectrum")
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
    np.savetxt("spike_frequencies.txt", freqs_dtf_order[:location_positive_nyquist+1])
    np.savetxt("spike_amplitudes.txt", y_hits_dtf_order[:location_positive_nyquist+1])
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
if not feed_into_second_pipe:
    # plt.errorbar(freqs, post_s_xi_mean, yerr=post_s_xi_std, color="green", ecolor="lightgreen", label=r"Posterior harmonic $\xi_s$", zorder=1)
    plt.plot(freqs, post_s_xi, "r-", label=r"Posterior harmonic $\xi_s$", zorder=2)
    usual_plot(xl="Frequency $f$", yl=r"$\xi_s$")


plot_wigner=False
if plot_wigner:

    ### ---- Calculation of Wigner function

    wigner_exists = False
    try:
        wigner_result = unpickle_me_this("wigner_result_pipe_1.pickle")

    except FileNotFoundError:

        fft_helper = ift.FFTOperator(space=0, domain=xi_subdomain)

        real_xi =  fft_helper(post_s_xi_mean)

        cutter = inference_scheme_pipe_1.X
        masker = inference_scheme_pipe_1.R_physical

        xi_real_cut_and_masked = masker(cutter.adjoint(real_xi))

        wigner_result  = Stress(xi_real_cut_and_masked)
        pickle_me_this("wigner_result_pipe_1", wigner_result)


    wigner_mat, t_dual, freqs = wigner_result


    ## ---- Print loud frequencies ( = rows/y-axis in imatshow plot)

    wr = wigner_mat.real
    loud_threshhold = 1e7
    loud_idcs = np.where(wr > loud_threshhold)[0]
    loud_freqs = np.unique(freqs[loud_idcs])

    print(f"Loud (stress > {loud_threshhold}) frequencies are : ",loud_freqs)


    ### ---- Set the DC-strip to 0 which contains very strong auto-terms and positively interfering cross-terms (even larger)

    to_cut = (-50 < freqs) & (freqs < 50)  # cut out the loud interference part at the 0 frequency...
    wigner_mat = wr.copy()
    wigner_mat[to_cut,:] = 0

    ### ---- Finally visualize the wigner function and a frequency-marginalized version

    visualize_stress(wigner_mat, rows=freqs, cols=t_dual+time[0])

    # marginalized = np.sum(wigner_mat, axis=0)
    # plt.plot(t_dual+time[0], marginalized)
    # usual_plot(yl="Stress", title="Frequency-Marginalized stress evolution")

if feed_into_second_pipe:

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


    print("\n\n\n\nEntering second pipe...\n")