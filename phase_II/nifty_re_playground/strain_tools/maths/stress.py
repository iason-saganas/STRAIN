import numpy as np
from scipy.interpolate import interp1d
import jax.numpy as jnp
from scipy.signal import find_peaks

import nifty.nifty.cl as ift
from typing import Literal

from ..basics.common_utils import raise_warning

__all__ = ["interpolate_waveform_from_inverted_wigner", "Stress_jft", "Stress_ift", "get_peaks",
                 "solve_data_equation_for_xi"]


def interpolate_waveform_from_inverted_wigner(new_times):
    waveform = np.loadtxt("/data/data_txt/waveform_from_inverted_wigner.txt")
    waveform_times = np.loadtxt("/data/data_txt/times_from_inverted_wigner.txt")

    # time_shift = waveform_times[0] - new_times[0]

    new_grid = np.linspace(waveform_times.min(), waveform_times.max(), len(new_times))
    interpolator = interp1d(x=waveform_times, y=waveform, fill_value="extrapolate")  # linear interpolation
    new_values = interpolator(new_grid)

    dt = new_times[1] - new_times[0]
    shift = int((0.136-0.25) /dt)  # please don't use this roll
    new_values = np.roll(new_values, -shift)

    # plt.plot(waveform_times, waveform, label="old")
    # plt.plot(new_times, new_values, label="new")
    # usual_plot()

    return new_values



def Stress_jft(xi, time, supress_print=False, downsample=False, norm="ortho"):
    """
    Implements S_ft, i.e. rows are frequencies and columns are times.

    See also nifty8 `Stress` function.

    :param downsample:
    :param xi: jnp.array        A field to calculate the wigner function for. Either of complex or real data type.
                                If complex, assumed to be in DFT standard order (DC first, then positives then negatives).
    :param time: jnp.array      The real-space time array at which xi (or its iFFT if complex) was sampled at.
    :param supress_print: bool, Print imaginary part diagonstics (Wigner function should be real).
    :return:
    """

    t0 = time[0]
    dt = time[1]-time[0]
    N = len(xi)
    f = jnp.fft.fftfreq(N, d=dt)
    k = f.copy()
    df = f[1] - f[0]
    t = jnp.arange(N) / (N*df)  # dual time, equal to input time - time[0].
    T = N * dt

    FFT_physical = lambda x, ax=-1: jnp.fft.fft(x, norm=norm, axis=ax) * T / jnp.sqrt(N)
    iFFT_physical = lambda x, ax=-1: jnp.fft.ifft(x, norm=norm, axis=ax) * jnp.sqrt(N) / T

    if jnp.iscomplexobj(xi):
        xi = iFFT_physical(xi)  # go to real space

    if downsample:
        step = 2
        xi = xi[::step]
        # time = time[::step]

    if not supress_print:
        print("\nCalculating stress...")

    t_c = t[:, None]  # time cast
    k_c = k[None, :]  # shift frequencies cast
    xi_c = xi[:, None]  # xi values cast as rows

    if not supress_print:
        print("\t Calculating zeta plus")
    zeta_plus = jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    if not supress_print:
        print("\t Calculating zeta minus")
    zeta_minus = jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    if not supress_print:
        print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT_physical(zeta_plus, ax=0)

    if not supress_print:
        print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT_physical(zeta_minus, ax=0)

    if not supress_print:
        print("\t Calculating Phi matrix")
    Phi = tilde_zeta_plus * tilde_zeta_minus.conj()  # domain = (h_space, h_space)

    if not supress_print:
        print("\t Inverse Fourier-Transforming columns of Phi matrix")
    S = iFFT_physical(Phi, ax=1)
    S.block_until_ready()

    if not supress_print:
        print("\t ... Done")
    if not supress_print:
        diagnostic = jnp.abs(jnp.mean(S.imag))
        tmp = float(diagnostic)
        if diagnostic < 1e-10:
            print(f"\u2714 Mean imaginary part of stress field is smaller than 1e-10 threshold ({diagnostic}) ")
        else:
            raise_warning(
                f"Realness threshold was not passed. Mean imaginary part of stress field larger than 1e-10 ({diagnostic}).")
    return S, t+t0, f


def fieldify(array, dom):
    return ift.Field(dt_(dom), array)


def dt_(dom):
    return ift.DomainTuple.make(dom)


def Stress_ift(xi_field: ift.Field, supress_print=False):
    """

    DFT conventions:
        F[m] = sum_{n=0}^{N-1} f[n] * exp(2πi * m n / N)
        f[n] = (1/N) * sum_{m=0}^{N-1} F[m] * exp(-2πi * m n / N)

    :param xi_field:  ift.Field    The Field over real space to analyze. If harmonic, assumed to be in DFT standard order and is mapped to its real space counterpart.
    :param supress_print:          Whether to print imaginary part diagnostic.
    :return: (S_mat, t_dual, f)    S_mat is the calculated wigner matrix, t_dual is the dual time from fourier transforming
                                   the columns of an intermediate matrix in ascending, monotonic order and f are the
                                   frequencies of the setup in standard DFT order (i.e. 0 as the first element).

    """

    if xi_field.domain[0].harmonic:
        helper_fft = ift.FFTOperator(xi_field.domain, space=0)
        xi_time_field = helper_fft(xi_field)
    else:
        xi_time_field = xi_field


    time_space = xi_time_field.domain[0]
    h_space = time_space.get_default_codomain()

    N = time_space.size
    t_vol = time_space.scalar_dvol
    h_vol = h_space.scalar_dvol * N

    dt_step = time_space.distances[0]
    xi_time = xi_time_field.val.astype(np.complex128)
    N = len(xi_time)
    f = np.fft.fftfreq(N, d=dt_step)
    k = f.copy()
    t = np.arange(N)*dt_step
    time = np.arange(N)*dt_step

    FFT_1 = ift.FFTOperator(domain=(time_space, h_space), space=0) * (1/t_vol)
    FFT_2 = ift.FFTOperator(domain=(h_space, h_space), space=1) * (1/h_vol)

    time_cast = time[:, None]
    k_freq_cast = k[None, :]
    xi_values_cast_as_rows = xi_time[:, None]

    print("\nCalculating stress...")

    print("\t Calculating zeta plus")
    zeta_plus = fieldify(np.exp(-np.pi*k_freq_cast*1j*time_cast) * xi_values_cast_as_rows, dom=(time_space,h_space))
    print("\t Calculating zeta minus")
    zeta_minus = fieldify(np.exp(np.pi*k_freq_cast*1j*time_cast) * xi_values_cast_as_rows, dom=(time_space,h_space))

    print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT_1(zeta_plus).val
    print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT_1(zeta_minus).val

    print("\t Calculating Phi matrix")
    Phi_val = tilde_zeta_plus * tilde_zeta_minus.conj()  # im putting the conjugate on the MINUS zeta since I also changed FFT convention by a sign wrt. Wikipedia...
    Phi_field = ift.Field(dt_((h_space, h_space)), val=Phi_val)

    print("\t Fourier-Transforming columns of Phi matrix")
    S = FFT_2(Phi_field)
    S_mat = S.val
    print("\t ... Done")

    dk = k[1] - k[0]        # safe in FFT ordering (first step is Δk)
    dt_dual = 1.0 / (N * dk)
    t_dual = np.arange(N) * dt_dual

    if not supress_print:
        diagnostic = np.abs(np.mean(S_mat.imag))
        if diagnostic < 1e-10:
            print(f"\u2714 Mean imaginary part of stress field is smaller than 1e-10 threshold ({diagnostic}) ")
        else:
            raise_warning(
                f"Realness threshold was not passed. Mean imaginary part of stress field larger than 1e-10 ({diagnostic}).")

    return S_mat, t_dual, f


def DEPR_get_peaks_from_cache(mode=Literal["global mean threshold", "scipy", "rolling mean threshold"], only_positives = True, thresh=2,
                         custom_norm=1, power_spectrum=None, custom_path=None, make_real_callable=lambda p: p.real):
    """
    Loads a saved xi_d file, detects peaks based on an ad-hoc threshhold and returns the position and amplitude of
    said peaks.
    :param mode:
    :param only_positives:      bool,        Whether to report events at negative frequencies
    :param thresh:        float,       Threshold in standard deviations or prominence.
    :param custom_norm:         float,       Normalization factor, e.g. max(posterior_pipe_1_ps_mean). By default,
                                             max(amplitude of peak) = 1.
    :param custom_path:                      Path to a saved xi_d file
    :param power_spectrum:                   If not none, will be used as weights
    :return:
    """
    path = "pipe2_xi_cache.txt" if custom_path is None else custom_path
    obj = np.loadtxt(path, dtype=np.complex128)

    print(f"...Trying to search for positive peaks (True) and negative peaks ({not only_positives})")

    xi, f = make_real_callable(obj[:,0]), obj[:,1].real
    if power_spectrum is None:
        power_spectrum = np.ones(len(f))
    if only_positives:
        to_del = np.where(xi<0)
        xi = np.delete(xi, to_del)
        f = np.delete(f, to_del)
        ps = np.delete(power_spectrum, to_del)


    if mode == "global mean threshold":
        print("\tThis mode does not support finding negative peaks")
        sigma_thresh = thresh
        adhoc_treshhold = sigma_thresh * np.mean(xi)
        print("\t\t...Finding peaks in xi larger than ", adhoc_treshhold)
        where_peaks_in_xi = np.where(xi > adhoc_treshhold)
    elif mode == "scipy":
        print("\tI'm not sure if this mode supports finding negative peaks, I think so")
        prom = thresh
        where_peaks_in_xi, properties = find_peaks(xi, prominence=prom)
    elif mode == "rolling mean threshold":
        print("\tThis mode supports finding negative peaks")
        window_length = 20  # Hz, window length
        df = f[1]-f[0]
        idx_half_length = int(window_length/2/df)
        N = len(xi)
        where_peaks_in_xi = []

        for idx_position in range(idx_half_length, N-idx_half_length):
            left_idx = idx_position - idx_half_length  # starts at 0 on the left
            right_idx = idx_position + idx_half_length + 1  # and  2*idx_half_length + 1 on the right
            # total size: right - left = 2*idx_half_length + 1
            # ends at: N-idx_half_length - 1 - idx_half_length = N - 1 - 2*idx_half_length on the left
            # and N - idx_half_length - 1 + idx_half_length + 1 = N
            # total size: right - left = N - N + 1 + 2*idx_half_length = 2*idx_half_length + 1
            xi_subslice = xi[left_idx:right_idx]

            mean_sub_xi = np.mean(xi_subslice)
            sig_sub_xi = np.std(xi_subslice)


            sigma_threshhold_plus = mean_sub_xi + thresh * sig_sub_xi
            sigma_threshhold_minus = mean_sub_xi - thresh * sig_sub_xi
            # mean_threshhold = mean_sub_xi * thresh

            if xi[idx_position] > sigma_threshhold_plus and xi[idx_position] == np.max(xi_subslice):
                where_peaks_in_xi.append(idx_position)

            if xi[idx_position] < sigma_threshhold_minus and xi[idx_position] == np.min(xi_subslice):
                where_peaks_in_xi.append(idx_position)

        where_peaks_in_xi = np.array(where_peaks_in_xi)

    else:
        raise ValueError("Mode must be 'constant' or 'rolling average'")

    peaks_k = f[where_peaks_in_xi]
    amplitudes_k = xi[where_peaks_in_xi]

    if custom_norm:
        ps_weights_normed = custom_norm
    else:
        # functional weights
        ps_weights = ps[where_peaks_in_xi]
        ps_weights_normed = ps_weights / max(ps_weights)

    # normed_amplitudes_k = amplitudes_k / max(amplitudes_k) * ps_weights_normed
    normed_amplitudes_k = amplitudes_k / max(amplitudes_k) * ps_weights_normed

    return peaks_k, normed_amplitudes_k


def get_peaks(local_sigma_threshold=3, global_sigma_threshold=3, window_length=20,
                            take_abs_of_amplitudes=False, custom_amplitude_norm=1, custom_path=None,
                            custom_xi_and_f=None):
    """
    Improved functionality, use in new code.

    Moving average sigma treshold detection: Checks whether the center of a 'window_length'-length window is above
    or below average by mu ± sigma_threshold * local std.

    If yes, the point is flagged as a candidate.
    Condition for acceptance: non-gaussianity induced by peak sticking out.

    The window is then moved by 1 pixel to the right.

    :param local_sigma_threshold:     Sigma thresh for local sliding window
    :param global_sigma_threshold:    Global thresh for very large peaks
    :param window_length:       The physical window length in Hz.
    :param custom_amplitude_norm:   Is multiplied onto the normed amplitudes
    :param custom_xi_and_f:     A list of your custom xi and f.
    :param custom_path:         If custom_xi_and_f is None, reads out xi and f information from file at this location
    :return:
    """
    print(f"Searching search for positive and negative peaks in the real part of xi")

    if custom_xi_and_f is None:
        path = "pipe2_xi_cache.txt" if custom_path is None else custom_path
        obj = np.loadtxt(path, dtype=np.complex128)
        xi, f = (obj[:, 0]).real, obj[:, 1].real
    else:
        xi = custom_xi_and_f[0]
        f = custom_xi_and_f[1]

    power_spectrum = np.ones(len(f))
    print("\tThis mode supports finding negative peaks")
    window_length = 20  # Hz, window length
    df = f[1] - f[0]
    idx_half_length = int(window_length / 2 / df)
    w = idx_half_length
    N = len(xi)
    where_peaks_in_xi = []

    # Global peaks
    sigma_threshhold_plus = np.mean(xi) + global_sigma_threshold * np.std(xi)
    sigma_threshhold_minus = np.mean(xi) - global_sigma_threshold * np.std(xi)
    for i, xi_value in enumerate(xi):
        if xi_value > sigma_threshhold_plus:
            where_peaks_in_xi.append(i)
        if xi_value < sigma_threshhold_minus:
            where_peaks_in_xi.append(i)


    def gaussian_fraction_check(some_xi, tol=0.05):
        # GPT, was too lazy
        mean = np.mean(some_xi)
        sigma = np.std(some_xi)
        within_1sigma = np.sum((some_xi >= mean - sigma) & (some_xi <= mean + sigma)) / len(some_xi)
        within_2sigma = np.sum((some_xi >= mean - 2 * sigma) & (some_xi <= mean + 2 * sigma)) / len(some_xi)
        return (abs(within_1sigma - 0.68) < tol) and (abs(within_2sigma - 0.95) < tol)


    # Local sliding average peaks
    for i in range(w, N - w):
        left_idx = i - w  # starts at 0 on the left
        right_idx = i + w + 1  # and  2*idx_half_length + 1 on the right
        # total size: right - left = 2*idx_half_length + 1
        # ends at: N-idx_half_length - 1 - idx_half_length = N - 1 - 2*idx_half_length on the left
        # and N - idx_half_length - 1 + idx_half_length + 1 = N
        # total size: right - left = N - N + 1 + 2*idx_half_length = 2*idx_half_length + 1
        xi_subslice = xi[left_idx:right_idx]
        f_subslice = f[left_idx:right_idx]  # for debugging

        mean_sub_xi = np.mean(xi_subslice)
        sig_sub_xi = np.std(xi_subslice)

        sigma_threshhold_plus = mean_sub_xi + local_sigma_threshold * sig_sub_xi
        sigma_threshhold_minus = mean_sub_xi - local_sigma_threshold * sig_sub_xi

        if xi[i] > sigma_threshhold_plus and xi[i] == np.max(xi_subslice) and not gaussian_fraction_check(xi_subslice):
            where_peaks_in_xi.append(i)

        if xi[i] < sigma_threshhold_minus and xi[i] == np.min(xi_subslice) and not gaussian_fraction_check(xi_subslice):
            where_peaks_in_xi.append(i)

        # debug_idx = jnp.argmin(jnp.abs(f-650))
        # if i == debug_idx:
        #     print("Debugging get_peaks_from_cache_v2")


    where_peaks_in_xi = np.array(where_peaks_in_xi)
    peaks_k = f[where_peaks_in_xi]
    amplitudes_k = xi[where_peaks_in_xi]

    if take_abs_of_amplitudes:
        amplitudes_k=jnp.abs(amplitudes_k)

    normed_amplitudes_k = amplitudes_k / max(amplitudes_k) * custom_amplitude_norm
    return peaks_k, normed_amplitudes_k


def solve_data_equation_for_xi(data, ps):
    r"""
    Calculates:

        xi = F(data)/sqrt{ps},

    which comes from the data model equation d = F^{-1} \sqrt{ps} xi. Uses jnp.fft.

    :param data:
    :param ps:
    :return:    Found harmonic xi that can be input into the Wigner function.
    """
    data_tilde = jnp.fft.fft(data)
    amp = jnp.sqrt(ps)
    return data_tilde / amp