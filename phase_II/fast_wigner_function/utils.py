import numpy as np
import matplotlib.pyplot as plt
import warnings

def xi_field(case, N=None, omegas=None, peak_frequency=None, peak_amplitude=None, base_elevation=None, idx=20, length=None,
             supress_print=False, use_complex=None):

    """

    Returns a Fourier xi field for some cases for which we have an analytic expression of the discretized stress field function
    over for which we can think about what characteristics should be visible in the stress field.

    For the analytical results I assume the discretized version of S used in this notebook.

    case 1:

        xi(ω) = δ_ωω⁎ * a => S(t,ω) = δ_ωω⁎ * a^2 * Δν / (2π) =  δ_ωω⁎ * a^2 *  2Δω/(2π) = δ_ωω⁎ * a^2 * 1/(πL)

        where in the last step we assumed that ν = 2ω was used for the dummy variable in the discretization scheme.

    case 4:
        <xi(ω) xi(ω')> -> δ_ωω' / Δω = δ_ωω' * L

        then, we expect (again assuming Δν = 2Δω) :

            <S(t, ω)> = 1/π ~ 0.318


    :param N:               How many support points in real space
    :param case:            1-4, see cases dict below.
    :param omegas:          The frequency values at which xi should be reported.
    :param peak_frequency:  The position of any peaks in fourier space.
    :param idx:             Extends the peak to a comb from peak_frequncy_idx till peak_frequncy_idx + idx iff case == 3.
    :param peak_amplitude:  The amplitude of any peaks in fourier space.
    :param base_elevation:  Iff case == 2 the base elevation in Fourier space next to the peak.
    :param length:          The real space length used to construct the correct dirac delta discrete representation for the covariance of standard normal variables.
    :param supress_print
    :param use_complex
    :return:
    """

    if omegas is not None:
        N = len(omegas)

    case_names = {
        1: "Spike",
        2: "Spike with constant non-zero elevation",
        3: "Frequency range",
        4: "Normal standard variable"
    }

    if not supress_print:
        print(f"\nConstructing xi field for case {case}: {case_names[case]}\n")

    if case == 1:
        ### --- case 1: Spike
        i_star = np.argmin(np.abs(omegas-peak_frequency))
        omega_star = omegas[i_star]

        print("\tf star: ", omega_star, " at index ", i_star)

        xi = np.zeros(N)
        xi[i_star] = peak_amplitude
    elif case == 2:
        ### --- case 2: Spike with base elevation
        i_star = np.argmin(np.abs(omegas-peak_frequency))
        omega_star = omegas[i_star]

        print("\tf star: ", omega_star, " at index ", i_star)

        xi = np.ones(N) * base_elevation
        xi[i_star] = peak_amplitude

    elif case == 3:
        ### --- case 3: Step function over frequency range
        i_star = np.argmin(np.abs(omegas-peak_frequency))
        omega_star = omegas[i_star]

        print("\tPeak elevation at: ", omega_star, " till ", omegas[i_star+idx])

        xi = np.zeros(N)
        xi[i_star:i_star+idx] = peak_amplitude
    elif case == 4:
        ### --- case 4: Step function over frequency range
        xi = sn_complex_xi(N, scale=length, use_complex=use_complex)
    else:
        raise ValueError("Unknown case")

    return xi


def sn_complex_xi(size, scale=1, use_complex=True):
    # standard normal, complex valued xi
    # scale sets the standard deviation! => scale' = sqrt(scale) if you want to set variance by scale
    if use_complex:
        res = 1/2*np.random.randn(size) + 1/2*np.random.randn(size) * 1j
    else:
        res = np.random.randn(size)
    return scale * res


def mirror_negative_frequencies(arr, standard_order:bool, unique_k_lengths=False):
    """
    Takes an array which is a function of fourier lengths and constructs the full array over the negative frequencies,
    assuming a real field. For example:

        nu = [0.  0.5 1.  1.5 2.  2.5]  -> [-2.5 -2.  -1.5 -1.  -0.5  0.   0.5  1.   1.5  2. ]

    and therefore correspondingly

        field = [ 0.1  0.4 -0.2  1.5 -0.5  1.1]  -> [1.1 -0.5 1.5 -0.2 0.4 0.1 0.4 -0.2 1.5 -0.5].

    If standard_order == True, the standard FFT order will be applied, i.e.: arr[0] corresponds to the 0 frequency. Then with ascending index come positive
    frequencies until the Nyquist frequency EXCLUDED, after which the negative frequencies come with their absolute value decreasing with ascending index.

    E.g.: nu = [0.  0.5 1.  1.5 2.  2.5] -> [ 0.   0.5  1.   1.5  2.  -2.5 -2.  -1.5 -1.  -0.5]

    Standard order source: https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
    Note that this is also very likely what harmonic_space.get_k_length_array() returns (or rather the absolute value of the standard order).

    :param standard_order:  Whether to apply the standard FFT order or not, in which the array will be truly mirrored around k=0.
    :param unique_k_lengths:            Whether the input array represents unique k-lengths. If not, a field is assumed.
    :param arr:             The array to mirror. This array must be reported at unique k-lengths! I.e. from nu=0 to nu=N_nyquist.
    :return:
    """
    if standard_order:
        return np.concatenate([[arr[0]], arr[1:-1], -arr[-1:0:-1]])
    if unique_k_lengths:
        to_the_left = -arr[1:][::-1]
    else:
        if standard_order:
            return
        to_the_left = arr[1:][::-1].conj()
    res = np.append(to_the_left, arr[:-1])
    return res


def visualize_stress(stress_matrix, rows, cols):
    plt.figure(figsize=(8,6))
    plt.imshow(stress_matrix, origin='lower', aspect='auto',
               extent=[np.min(cols), np.max(cols), np.min(rows), np.max(rows)],
               cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Stress')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [s]')
    plt.title('Time vs Frequency')
    plt.tight_layout()
    plt.show()


# --- map frequencies to DFT indices ---
def freq_to_index(f, omegas):
    omega_min = np.min(omegas)
    delta_omega = omegas[1] - omegas[0]
    N = len(omegas)
    # map f to nearest DFT bin index
    idx = np.round((f - omega_min)/delta_omega).astype(int) % N
    return idx

def raise_warning(msg):
    warnings.warn(msg, category=UserWarning, stacklevel=2)


class CallableArray:
    """Callable mapping from keys (e.g. frequencies) to array values."""

    def __init__(self, x, y, bounds_error=False, rtol=1e-12, atol=0.0,
                 bound_min=None, bound_max=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        sort_idx = np.argsort(x)
        self._x = x[sort_idx]
        self._y = y[sort_idx]

        self._bounds_error = bounds_error
        self._rtol = rtol
        self._atol = atol

        # default symmetric bounds if none are given
        self._bound_min = np.min(x) if bound_min is None else bound_min
        self._bound_max = np.max(x) if bound_max is None else bound_max

    def __call__(self, omega):
        omega = np.asarray(omega)
        out = np.zeros_like(omega, dtype=self._y.dtype)

        # out-of-bounds mask
        mask_oob = (omega < self._bound_min) | (omega > self._bound_max)
        if np.any(mask_oob) and self._bounds_error:
            bad_idx = np.argwhere(mask_oob)
            raise ValueError(
                f"Frequencies at indices {bad_idx.tolist()} "
                f"are out of bounds [{self._bound_min}, {self._bound_max}]"
            )

        # positions in sorted grid
        idx = np.searchsorted(self._x, omega)

        # exact matches at idx
        mask = (idx < len(self._x)) & np.isclose(self._x[np.clip(idx, 0, len(self._x) - 1)],
                                                 omega, rtol=self._rtol, atol=self._atol)
        out[mask] = self._y[np.clip(idx, 0, len(self._x) - 1)][mask]

        # exact matches at idx-1
        mask2 = (idx > 0) & np.isclose(self._x[np.clip(idx - 1, 0, len(self._x) - 1)],
                                       omega, rtol=self._rtol, atol=self._atol) & ~mask
        out[mask2] = self._y[np.clip(idx - 1, 0, len(self._x) - 1)][mask2]

        # check misses (in-grid but not matched)
        mask_miss = ~(mask | mask2 | mask_oob)
        if np.any(mask_miss):
            misses = omega[mask_miss]
            raise KeyError(f"Frequencies {misses[:10]}... not in grid (between points)")

        return out


def check_if_mat_imag(mat):
    diagnostic = np.abs(np.mean(mat.imag))
    if diagnostic < 1e-10:
        print(f"\u2714 Mean imaginary part of stress field is smaller than 1e-10 threshold ({diagnostic}) ")
    else:
        raise_warning(f"Realness threshold was not passed. Mean imaginary part of stress field larger than 1e-10 ({diagnostic}).")


def thin_out(arr, num):
    # takes away every second point num times
    for _ in range(num):
        arr = arr[::2]
    return arr