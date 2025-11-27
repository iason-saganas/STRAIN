import jax.numpy as jnp
import numpy as np
from nifty.nifty.re.conjugate_gradient import static_cg
from scipy.signal.windows import tukey
import operator
from jax.numpy import fft


def hartley(p, signal_grid):
    tmp = fft.fftn(p, axes=None)
    add = operator.add
    harmonic_dvol = 1.0 / signal_grid.total_volume
    return add(tmp.real, tmp.imag) * harmonic_dvol


def pseudo_inverse(A, A_T, b):
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

    :param A:                       A linear operator.
    :param A_T:                     The transpose of A.
    :param b:                       The data vector.
    :return:
    """

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

    return xi


def find_penrose_moore_solution(pipe,  reload_from_cache=True, filename="pipe2_xi_cache.txt"):
    """
    :param pipe:                    An instance of the InferenceSchemeRe class containing the data power spectrum
                                    stored as an inferred posterior mean power spectrum.
    :param reload_from_cache:       Whether to reload cached results under a 'pipe2_xi_cache.txt' file.
    :return:
    """

    if reload_from_cache:
        try:
            return np.loadtxt(filename, dtype=np.complex128)[:,0]
        except FileNotFoundError:
            print(f"File {filename} not found, recalculating xi")
            pass
    else:
        print("Calculating penrose moore xi")

    window_function = tukey(M=pipe.n_ds, alpha=0.1, sym=True)
    d = (window_function * pipe.d).astype(jnp.complex128)

    h_trafo = lambda p: jnp.fft.fft(p, norm="ortho")
    h_inv_trafo = lambda p: jnp.fft.ifft(p, norm="ortho")

    ps_mean_std, _, _ = pipe.get_posterior_statistics()
    ps_mean = (ps_mean_std[0])[pipe.s_h_dom_expander]
    M, N = pipe.n_ss, pipe.n_ds

    A = lambda p: sample_from_ps(p, N, ps_mean, h_inv_trafo)
    A_T = lambda p: sample_from_ps_transpose(p, ps_mean, M, N, h_trafo)

    xi = pseudo_inverse(A=A, A_T=A_T, b=d)

    if reload_from_cache:

        to_save = np.column_stack((xi, pipe.k_signal_full))
        np.savetxt(filename, to_save)

    return xi


def sample_from_ps(xi, N, ps, inverse_h_trafo):
    r"""
    Implements A = mask @ F^{-1} @ \sqrt{ps} for d = A*xi.

    :param xi:                  The vector to solve for.
    :param N:                   The number of data points.
    :param ps:                  The power spectrum used for the generation of data points.
    :param inverse_h_trafo:     A unitary ifft.
    :return:
    """
    amp = jnp.sqrt(ps)
    res = amp * xi
    return inverse_h_trafo(res)[:N]


def sample_from_ps_transpose(d, ps, M, N, h_trafo):
    r"""
    The transpose of `sample_from_ps`, i.e.

        A^T = \sqrt{ps}^T @ F^{-1}^T @ mask^T

    For F, the Hartley transform is used, which is hopefully orthogonal such that F^{-1}^T = F. Since
    ps is diagonal, \sqrt{ps}^T = \sqrt{ps}, so the formula reduces to

        A^T = \sqrt{ps} @ F @ mask^T

    For now, the data is tapered, but this does not need to be done if a tapering function instead of a simple
    "cutter" is employed in the forward model.

    :param d:               The data vector.
    :param ps:              The power spectrum used for the generation of data points.
    :param N:               The number of data points.
    :param M:               The number of points in signal space.
    :param h_trafo:         A unitary fft.
    :return:
    """

    mask_T_d = jnp.concatenate((d, jnp.zeros(M-N)))
    amp = jnp.sqrt(ps)
    res = amp * h_trafo(mask_T_d)
    return res