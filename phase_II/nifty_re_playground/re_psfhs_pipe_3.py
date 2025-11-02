from re_psfhs_pipe_1 import *
from scipy.signal.windows import tukey
from nifty.nifty.re.conjugate_gradient import static_cg

window_function = tukey(M=len(time), alpha=0.1, sym=True)
strain_tapered = window_function * strain

data_grid = pipe_1.d_dom_real
harmonic_signal_grid = pipe_1.s_dom_harmonic
signal_distributor = harmonic_signal_grid.power_distributor

# d_tilde = hartley(strain_tapered, signal_grid=data_grid)

posterior_pipe_1_ps_mean_std, _, _ = pipe_1.get_posterior_statistics()
posterior_pipe_1_ps_mean = (posterior_pipe_1_ps_mean_std[0])[signal_distributor]
N = pipe_1.n_ds
M = pipe_1.n_ss

# pipe_3_xi_s = d_tilde / np.sqrt(posterior_pipe_1_ps_mean)

# plt.plot(pipe_1.k_data_full, pipe_3_xi_s)
# plt.show()


def find_penrose_moore_solution(A, A_T, b):
    r"""

    Why this? Because we infer a denser power spectrum compared to the actual data power spectrum resolution.
    So now we want to find peaks => We should keep the same resolution => The dimensions definitely don't match.
    If you force the dimensions to match, that means you don't extend in signal space, leading to periodic boundary
    conditions.

    For simplicity, here, I assume periodic data through tapering; By including an asymmetric tapering function on the
    extended signal domain in the forward model, this assumption can even be lifted, I think. But what's really important
    is the location of the peaks.

    Let x be a vector of length M and b be a vector of length N, where M > N.
    Then, the system

        Ax = b

    where A is a linear operator is underdetermined (not enough data points to constrain all variables x_i).
    The Penrose-Moore solution is given by

        x = A^{⊕}b,

    where A^{⊕} is the pseudo-inverse and its given by

        A^{⊕} = A^T (A A^T)^{-1}.

    In the first step, the inverse of aa_T := A @ A^T must be applied onto the vector b. The inverse of aa_T exists,
    such that one can find b_prime:

         aa_T @ b_prime = b

    via the CG method. In the final step, A_T is applied onto b_prime, completing the pseudo-inversion.

    :param A:       A linear operator.
    :param A_T:     The transpose of A.
    :param b:       The data vector.
    :return:
    """
    try:
        return np.loadtxt("pipe3_xi_cache.txt", dtype=np.complex128)
    except FileNotFoundError:
        pass

    aa_T = lambda p: A(A_T(p))

    cg = static_cg
    max_iterations = 10_000  # to get a good fit to the data (at least visually, calculate the least squares for your
    # self one time)
    # max_iterations = 1000
    b_prime, cg_info = cg(aa_T, b, name="penrose_moore_cg", absdelta=1e-10, maxiter=max_iterations)  # signature:
    # mat, j where mat(x) = j is solved for x

    if cg_info is not None and cg_info < 0:
        raise ValueError("conjugate gradient failed")

    xi = A_T(b_prime)

    np.savetxt("pipe3_xi_cache.txt", xi)
    return xi


partial_hartley = lambda p: hartley(p, pipe_1.s_dom_real)
# h_trafo = partial_hartley
# h_inv_trafo = partial_hartley
h_trafo = lambda p: jnp.fft.fft(p, norm="ortho")  # So the A operator is truly unitary, don't know how partial_hartley
# treats volume factors
h_inv_trafo = lambda p: jnp.fft.ifft(p, norm="ortho")

def sample_from_ps(xi):
    r"""
    Implements A = mask @ F^{-1} @ \sqrt{ps} for d = A*xi.
    :param xi:
    :return:
    """
    amp = jnp.sqrt(posterior_pipe_1_ps_mean)
    res = amp * xi
    return h_inv_trafo(res)[:N]


def sample_from_ps_transpose(d):
    r"""
    The transpose of `sample_from_ps`, i.e.

        A^T = \sqrt{ps}^T @ F^{-1}^T @ mask^T

    For F, the Hartley transform is used, which is hopefully orthogonal such that F^{-1}^T = F. Since
    ps is diagonal, \sqrt{ps}^T = \sqrt{ps}, so the formula reduces to

        A^T = \sqrt{ps} @ F @ mask^T

    For now, the data is tapered, but this does not need to be done if a tapering function instead of a simple
    "cutter" is employed in the forward model.

    :param d:
    :return:
    """

    mask_T_d = jnp.concatenate((d, jnp.zeros(M-N)))
    amp = jnp.sqrt(posterior_pipe_1_ps_mean)
    res = amp * h_trafo(mask_T_d)
    return res

penrose_xi = find_penrose_moore_solution(A=sample_from_ps, A_T=sample_from_ps_transpose, b=strain_tapered.astype(jnp.complex128))

positive_penrose_xi = penrose_xi.real[penrose_xi.real > 0]
adhoc_treshhold = 2 * np.mean(positive_penrose_xi)
where_peaks_in_xi = np.where(penrose_xi > adhoc_treshhold)

peaks_k =  (pipe_1.k_signal_full)[where_peaks_in_xi]
amplitudes_k = penrose_xi[where_peaks_in_xi].real
norm_amplitudes_k = max(posterior_pipe_1_ps_mean)/max(amplitudes_k)

# Plot of penrose_xi and its 2 sigma peaks
plt.plot(pipe_1.k_signal_full, penrose_xi.real)
plt.plot(peaks_k, [adhoc_treshhold]*len(peaks_k), "r.", markersize=5)
plt.show()

_, k_lengths, power_spectrum = unpickle_me_this(
                    "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/results_from_welch_averaging_data.pickle",
                    absolute_path=True)
k_lengths = k_lengths[1:]  # remove 0-mode for simplicity
spectrum_welch = power_spectrum.val[1:]

# Plot of smooth background together with found peaks
plt.plot(pipe_1.k_signal_full, posterior_pipe_1_ps_mean, label=r"Smooth background $p_s(k)$")
plt.plot(peaks_k, norm_amplitudes_k, "b.", markersize=5, label=r"Normalized amplitudes of peaks in penrose $\xi$")
plt.plot(k_lengths, spectrum_welch, label=r"Empirical estimate of $p(k)$", color="orange")
plt.legend()
plt.loglog()
plt.show()

# Plot in data space
posterior_penrose_data = sample_from_ps(penrose_xi)
plt.plot(time, posterior_penrose_data.real, label="Penrose-Moore")
plt.plot(time, strain_tapered, label="Tapered data")
plt.legend()
plt.show()


