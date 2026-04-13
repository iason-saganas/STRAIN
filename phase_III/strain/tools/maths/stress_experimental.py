import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from ..basics.common_utils import raise_warning

__all__ = ["Stress_jft_experimental"]


def boundary_differences(M):
    """
    M: 2D array (can be complex)
    returns:
        col_diff, row_diff  (complex arrays)
    """
    M = np.asarray(M)

    # columns: last row - first row
    col_diff = M[-1, :] - M[0, :]

    # rows: last column - first column
    row_diff = M[:, -1] - M[:, 0]

    return col_diff, row_diff

def plot_boundary_differences(M):
    col_diff, row_diff = boundary_differences(M)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # columns
    axs[0].plot(col_diff.real, label="Re(col diff)")
    axs[0].plot(col_diff.imag, label="Im(col diff)")
    axs[0].set_title("Column boundary differences")
    axs[0].set_xlabel("Column index")
    axs[0].legend()

    # rows
    axs[1].plot(row_diff.real, label="Re(row diff)")
    axs[1].plot(row_diff.imag, label="Im(row diff)")
    axs[1].set_title("Row boundary differences")
    axs[1].set_xlabel("Row index")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def pad_matrix(M, extent=0.1):
    """
    Pad a square matrix on all sides with its boundary values.

    Parameters
    ----------
    M : 2D square array
    extent : float
        Fraction of matrix size to pad on each side

    Returns
    -------
    M_padded : 2D array
        Padded matrix
    """
    M = np.asarray(M)
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")

    n = M.shape[0]
    pad = int(n * extent)

    print("Padding with ", pad, " elements, corresponding to ", np.round(100*pad/n,2), "% of array length")

    # top/bottom padding: repeat first/last row
    top = np.repeat(M[0:1, :], pad, axis=0)
    bottom = np.repeat(M[-1:, :], pad, axis=0)

    M_vert = np.vstack([top, M, bottom])

    # left/right padding: repeat first/last column
    left = np.repeat(M_vert[:, 0:1], pad, axis=1)
    right = np.repeat(M_vert[:, -1:], pad, axis=1)

    M_padded = np.hstack([left, M_vert, right])
    return M_padded

def smooth_zero_pad_core(M, extent=0.1, alpha=1, beta=.5):
    """
    Pad a square matrix with its boundary values and taper only the padding toward zero.
    Inner core remains untouched.

    Parameters
    ----------
    M : 2D square array
    extent : float
        Fraction of matrix size used as padding

    Returns
    -------
    M_smooth : 2D array
        Padded matrix with aggressively tapered edges
    """
    M = np.asarray(M)
    n = M.shape[0]
    pad = int(n * extent)
    if pad == 0:
        return M.copy()

    # Step 1: pad with boundary values
    M_pad = pad_matrix(M, extent=extent)
    N = M_pad.shape[0]

    # Step 2: create taper windows for padding region only
    wx = np.ones(N)
    wy = np.ones(N)

    # Left/right taper
    if pad > 0:
        wx[:pad] = (beta * (1 - np.cos(np.pi * np.linspace(0, 1, pad))))**alpha
        wx[-pad:] = (beta * (1 - np.cos(np.pi * np.linspace(1, 0, pad))))**alpha
        wy[:pad] = (beta * (1 - np.cos(np.pi * np.linspace(0, 1, pad))))**alpha
        wy[-pad:] = (beta * (1 - np.cos(np.pi * np.linspace(1, 0, pad))))**alpha

    # Step 3: make 2D window
    window2d = wy[:, None] * wx[None, :]

    # Step 4: preserve inner core
    inner_slice = slice(pad, N-pad)
    M_smooth = M_pad.copy()
    # multiply only padding regions
    # top
    M_smooth[:pad, :] *= window2d[:pad, :]
    # bottom
    M_smooth[-pad:, :] *= window2d[-pad:, :]
    # left
    M_smooth[pad:-pad, :pad] *= window2d[pad:-pad, :pad]
    # right
    M_smooth[pad:-pad, -pad:] *= window2d[pad:-pad, -pad:]

    col_diff, row_diff = boundary_differences(M_smooth)

    if np.any(np.round(np.sum(col_diff.real), 12) != 0) or \
       np.any(np.round(np.sum(col_diff.imag), 12) != 0) or \
       np.any(np.round(np.sum(row_diff.real), 12) != 0) or \
       np.any(np.round(np.sum(row_diff.imag), 12) != 0):
        raise ValueError("Boundary differences are nonzero; increase extent to reduce leakage.")

    return M_smooth

def Stress_jft_experimental(xi, time, padding_extent=.1, supress_print=False, downsample=False, norm="ortho", tukey_window_where_necessary=False,
                               zp_func=smooth_zero_pad_core, debug_plot=False):
    """
    Implements S_ft, i.e. rows are frequencies and columns are times.

    See also nifty8 `Stress` function.

    :param xi: jnp.array        A field to calculate the wigner function for. Either of complex or real data type.
                                If complex, assumed to be in DFT standard order (DC first, then positives then negatives).
    :param time: jnp.array      The real-space time array at which xi (or its iFFT if complex) was sampled at.
    :param supress_print: bool, Print imaginary part diagonstics (Wigner function should be real).
    :return:
    """

    t0 = time[0]
    dt = time[1]-time[0]

    # extent = 1 + padding_extent
    N = len(xi) # * extent
    f = jnp.fft.fftfreq(N, d=dt)
    k = f.copy()
    df = f[1] - f[0]
    t = jnp.arange(N) / (N*df)  # dual time, equal to input time - time[0].
    T = N * dt

    FFT_physical = lambda x, ax=-1: jnp.fft.fft(x, norm=norm, axis=ax) * T / jnp.sqrt(N)
    iFFT_physical = lambda x, ax=-1: jnp.fft.ifft(x, norm=norm, axis=ax) * jnp.sqrt(N) / T

    if jnp.iscomplexobj(xi):
        print("INVERSE FOURIER TRANSFORMING xi")
        xi = iFFT_physical(xi)  # go to real space
    else:
        print("not INVERSE FOURIER TRANSFORMING xi")

    if downsample:
        step = 2
        xi = xi[::step]
        time = time[::step]

    if not supress_print:
        print("\nCalculating stress...")

    t_c = t[:, None]  # time cast
    k_c = k[None, :]  # shift frequencies cast
    xi_c = xi[:, None]  # xi values cast as rows

    if not supress_print:
        print("\t Calculating zeta plus")
    if tukey_window_where_necessary:
        # plot_boundary_differences(jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c)
        # zeta_plus = tukey_window_matrix(jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c, ax=0) # domain = (time_space, h_space)
        zeta_plus = zp_func(jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c, padding_extent) # domain = (time_space, h_space)
        if debug_plot:
            plot_boundary_differences(zeta_plus)
    else:
        zeta_plus = jnp.exp(-jnp.pi * k_c * 1j * t_c) * xi_c # domain = (time_space, h_space)

    if not supress_print:
        print("\t Calculating zeta minus")

    if tukey_window_where_necessary:
        # plot_boundary_differences(jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c)
        # zeta_minus = tukey_window_matrix(jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c, ax=0) # domain = (time_space, h_space)
        zeta_minus = zp_func(jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c, padding_extent) # domain = (time_space, h_space)
        if debug_plot:
            plot_boundary_differences(zeta_minus)
        # stop
    else:
        zeta_minus = jnp.exp(jnp.pi * k_c * 1j * t_c) * xi_c  # domain = (time_space, h_space)

    if not supress_print:
        print("\t Calculating zeta plus in Fourier space")
    tilde_zeta_plus = FFT_physical(zeta_plus, ax=0)

    if not supress_print:
        print("\t Calculating zeta minus in Fourier space")
    tilde_zeta_minus = FFT_physical(zeta_minus, ax=0)

    if not supress_print:
        print("\t Calculating Phi matrix")
    if tukey_window_where_necessary:
        # plot_boundary_differences(tilde_zeta_plus * tilde_zeta_minus.conj())
        # Phi = tukey_window_matrix(tilde_zeta_plus * tilde_zeta_minus.conj(), ax=1)  # domain = (h_space, h_space)
        print("NOT TUKEYING PHI, since it looks like in all cases its almost periodic...")
        Phi = tilde_zeta_plus * tilde_zeta_minus.conj()  # domain = (h_space, h_space)
        if debug_plot:
            plot_boundary_differences(Phi)
    else:
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

    if tukey_window_where_necessary:
        pad = int(len(xi) * padding_extent)
        N_pad = len(xi) + 2*pad
        f_pad = jnp.fft.fftfreq(N_pad, d=dt)
        df_pad = f[1] - f[0]
        t_pad = jnp.arange(N_pad) / (N_pad*df_pad)  # dual time, equal to input time - time[0].
        return S, t_pad+t0, f_pad
    else:
        return S, t+t0, f