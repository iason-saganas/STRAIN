import matplotlib.pyplot as plt

from power_spectrum_from_harmonic_stress_pipe_1 import *

# Get the gaussian comb operator. Initialized by a human!

assert np.allclose(np.diff(f), np.diff(f)[0], rtol=0, atol=1e-10)

df = np.diff(f)[0]

# (index of frequency at peak, amplitude of peak, half-width of the peak in INDEX counts)
information_idx = [(30, 15, 8), (3983, 1.64, 18), (5883, 1.1, 30)]

# => Convert idx_center and idx_width to freq_center and freq_width
information_freq = [( f[t[0]],  t[1], t[2] * df) for t in information_idx]
frequency_centers = [t[0] for t in information_freq]
peak_amplitudes = [t[1] for t in information_freq]
frequency_half_widths = [t[2] for t in information_freq]

h_dom = inference_scheme_pipe_1.domain_ext.get_default_codomain()
p_space = ift.PowerSpace(h_dom)
frequency_field = ift.Field(ift.DomainTuple.make(p_space, ), val=f)
gaussian_comb_op = generative_gaussian_comb(x_field=frequency_field, position_of_peaks=frequency_centers,
                                            amplitude_of_peaks=peak_amplitudes, half_width_of_peaks=frequency_half_widths,
                                            rel_sigma_lognormal=.5)

# Start with a fresh whiten noise harmonic xi_s
inference_scheme_pipe_2 = ExecuteEasyRGSpaceKL(
    discrete_time=time,
    d=data,
    cfm_model_name="s_",
    gaussian_noise_level=1e-10,
    out_dir_name="outs/power_spectrum_from_harmonic_stress_f_pipe_2/",
    fluct=(5, 2),
    llslope=(0, 1),
    custom_generative_model=None,
    kl_minimizations=10,
    op_to_apply_to_amp=gaussian_comb_op,
    x_fac=2,
    n_pix_fac=1
)

# inference_scheme_pipe_2.plot_power_spectrum_prior_samples(3)

inference_scheme_pipe_2.run()

# inference_scheme_pipe_2.plot_posterior(plot_signal_space=True)
# inference_scheme_pipe_2.plot_posterior(plot_signal_space=False)

k_domain_lengths, mean_amp_spec = inference_scheme_pipe_2.plot_posterior_pow_spec(show=False)


p_sls = inference_scheme_pipe_2.posterior_samples
post_mean, post_var = p_sls.sample_stat()

mdl = inference_scheme_pipe_2.model




### --- Little test, delete later

h_dom = mean_amp_spec.domain[0]._harmonic_partner
FT_to_real = ift.FFTOperator(domain=h_dom)
pd = ift.PowerDistributor(target=h_dom)
mean_amp_spec_full= pd(mean_amp_spec)
mean_amp_spec_full_inverse = mean_amp_spec_full.ptw("reciprocal")

s_m, _ = p_sls.sample_stat(mdl)  # posterior
# xi = ift.from_random(mean_amp_spec_full.domain)
# s_m = FT_to_real(mean_amp_spec_full * xi)  # from power spectrum itself

diag_op = ift.DiagonalOperator(mean_amp_spec_full_inverse)

s_m_tilde = FT_to_real.inverse(s_m)

whitened_signal = diag_op(s_m_tilde)
real_whitened_signal = FT_to_real(whitened_signal)
# whitened_data = inference_scheme_pipe_2.R_physical(real_whitened_signal)

data_from_posterior = inference_scheme_pipe_2.R_physical(inference_scheme_pipe_2.X.adjoint(s_m))
whitened_data_from_posterior = inference_scheme_pipe_2.R_physical(inference_scheme_pipe_2.X.adjoint(real_whitened_signal))

plt.plot(time, whitened_data_from_posterior.val, label="Whitened by posterior power spectrum")
plt.plot(time,data_from_posterior.val, label="Data from posterior")

plt.legend()
plt.show()


S_w = ift.power_analyze(whitened_signal)

print("mean and std: ", np.mean(S_w.val), np.std(S_w.val), " should both be approximately 1 if found power spectrum"
                                                           "is a match to empirical power spectrum")
plt.plot(S_w.val)
plt.xlabel("Frequency bin")
plt.ylabel("Normalized power")
plt.show()



stop


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


### ---- Plot posterior harmonix xi_s
# plt.errorbar(freqs, post_s_xi, yerr=post_s_xi_std.val, color="green", ecolor="lightgreen", label=r"Posterior harmonic $\xi_s$", zorder=1)
plt.plot(freqs, post_s_xi, "r-", label=r"Posterior harmonic $\xi_s$", zorder=2)
usual_plot(xl="Frequency $f$", yl=r"$\xi_s$")

plot_wigner=True
if plot_wigner:

    ### ---- Calculation of Wigner function

    wigner_exists = False
    try:
        wigner_result = unpickle_me_this("wigner_result_pipe_2.pickle")

    except FileNotFoundError:

        fft_helper = ift.FFTOperator(space=0, domain=xi_subdomain)

        real_xi =  fft_helper(post_s_xi_mean)

        cutter = inference_scheme_pipe_1.X
        masker = inference_scheme_pipe_1.R_physical

        xi_real_cut_and_masked = masker(cutter.adjoint(real_xi))

        wigner_result  = Stress(xi_real_cut_and_masked)
        pickle_me_this("wigner_result_pipe_2", wigner_result)


    wigner_mat, t_dual, freqs = wigner_result

    print("\tMean and std of wigner: ", np.mean(wigner_mat.real), np.std(wigner_mat.real))

    ## ---- Print loud frequencies ( = rows/y-axis in imatshow plot)

    # wr = wigner_mat.real
    # loud_threshhold = 1e7
    # loud_idcs = np.where(wr > loud_threshhold)[0]
    # loud_freqs = np.unique(freqs[loud_idcs])
    #
    # print(f"Loud (stress > {loud_threshhold}) frequencies are : ",loud_freqs)


    ### ---- Set the DC-strip to 0 which contains very strong auto-terms and positively interfering cross-terms (even larger)

    # to_cut = (-50 < freqs) & (freqs < 50)  # cut out the loud interference part at the 0 frequency...
    # wigner_mat = wr.copy()
    # wigner_mat[to_cut,:] = 0

    ### ---- Finally visualize the wigner function and a frequency-marginalized version

    visualize_stress(wigner_mat, rows=freqs, cols=t_dual+time[0])

    marginalized = np.sum(wigner_mat, axis=0)
    plt.plot(t_dual+time[0], marginalized)
    usual_plot(yl="Stress", title="Frequency-Marginalized stress evolution")