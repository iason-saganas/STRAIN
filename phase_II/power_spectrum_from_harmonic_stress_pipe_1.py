from phase_I.utils.config_jupyter_notebooks import *
from utils.helpers import *

data = 1e19 * strain.value
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

inference_scheme_pipe_1 = ExecuteEasyRGSpaceKL(
    discrete_time=time,
    d=data,
    cfm_model_name="s_",
    gaussian_noise_level=1e-10,
    out_dir_name="outs/power_spectrum_from_harmonic_stress_f_pipe_1/",
    fluct=(1,1),
    llslope=(0,1),
    custom_generative_model=None,
    kl_minimizations=20,
    x_fac=2,
    n_pix_fac=1
)

# response = inference_scheme._R_full
# rnd_data = response(ift.from_random(response.domain))
# np.savetxt("tmp_rnd_data.txt", rnd_data.val)

# inference_scheme_pipe_1.plot_power_spectrum_prior_samples(num=1)
# plt.show()

# plt.plot(time, data)
# inference_scheme_pipe_1.plot_prior_samples(num=1)
# plt.show()
# stop

inference_scheme_pipe_1.run()

if not feed_into_second_pipe:
    inference_scheme_pipe_1.plot_posterior(plot_signal_space=False)
    inference_scheme_pipe_1.plot_posterior(plot_signal_space=True)

k_domain_lengths, mean_pow_spec = inference_scheme_pipe_1.plot_posterior_pow_spec(show=not feed_into_second_pipe)


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


### ---- Get zero-centered posterior harmonic xi and frequencies for plotting

post_s_xi = hartley_to_fftshift(post_s_xi_mean.val)
freqs = hartley_to_fftshift(xi_subdomain[0].get_k_length_array().val, flip_negatives=True)


### ---- Manual peak detection using simple threshhold from zero-centered xi_s

threshhold_plus = 0.25
threshhold_minus = -np.inf
idx_hits_plus = np.where(post_s_xi >= threshhold_plus)
idx_hits_minus = np.where(post_s_xi <= threshhold_minus)

idx_hits = np.append(idx_hits_plus, idx_hits_minus).flatten()

y_hits = np.zeros(len(freqs))


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

    post_pipe_1_slope_mean = post_llslope[0]
    post_pipe_1_slope_std = post_llslope[1]

    post_pipe_1_fluct_mean = post_fluct[0]
    post_pipe_1_fluct_std = post_fluct[1]

    print("\n\n\n\nEntering second pipe...\n")