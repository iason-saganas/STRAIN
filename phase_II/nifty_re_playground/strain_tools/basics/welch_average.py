import jax.numpy as jnp
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .common_utils import fw_hartley
from .plotting import usual_plot

__all__ = ["calculate_welch_average"]

# def fw_hartley(x, norm="ortho"):
#     r"""
#     If ortho, preserves scaling of input. I.e.
#
#         jnp.var(fw_hartley(\xi)) = jnp.var(\xi) if e.g. \xi is iid.
#
#     :param x:
#     :param norm:
#     :return:
#     """
#     N = len(x)
#     Xf = jnp.fft.fft(x)  # Accumulates √N of intrinsic scaling
#     Hx = Xf.real - Xf.imag  # standard Hartley: cos+sin → real - imag
#     if norm == "ortho":
#         Hx = Hx / jnp.sqrt(N)  #  scales with 1/√N
#     return Hx  # total scale: 1 if ortho, else √N
#
#
# def bw_hartley(Hx, norm="ortho"):
#     r"""
#     This is unitary if ortho norm i.e.
#
#             xi = jnp.random.standard_normal(8193)
#             v = bw_hartley(xi)
#
#             v.T @ v ==  xi.T @ xi
#
#     :param Hx:
#     :param norm:
#     :return:
#     """
#     # Hartley is its own inverse! Note: H(H(x)) = N for not-normalized Hartley H.
#     # Further, Hx = fw_ortho_hartley(x) ~ 1.
#     N = len(Hx)
#     x = fw_hartley(Hx, norm=None)  # ~ √N if input scales with 1 (which it does if it comes from ortho fw hartley)
#     if norm == "ortho":
#         x = x / jnp.sqrt(N) # scales with 1/√N
#     return x  # total scale: 1 if ortho AND input is from ortho fw_hartley.
#     # if instead input is not from non-ortho fw_hartley: ~ √N I think.


def power_analyze_re_hartley(y_values):
    """
    Returns an estimate of the power spectrum by absolute squaring the fourier transform.
    :param y_values: A real space periodic array.
    :return:
    """
    return jnp.abs(fw_hartley(y_values, norm="ortho"))**2


def calculate_welch_average(x, y, L=2, leave_out=None, debug=False, output_on_full_harmonic_domain=False,
                            final_average_call=jnp.mean,
                            tapering_function=lambda d: tukey(M=len(d), alpha=0.1, sym=True)):
    """
    Subdivides data into little windows of length L, tapers, takes their fourier-transform absolutely squared 
    and performs an average over all windows.

    All outputs are always in DFT standard order, i.e. 0 frequency first, then positive then negative frequencies.

    The windows have no overlap.

    :param x:
                                A jnp.array of time values over which the data are sampled.
    :param y:
                                The data. jnp.array or ift.Field
    :param L:
                                The real space length of little windows to subdivide the data under.
    :param debug:               Show debug plots.
    :param leave_out:
                                A tuple (t_init, t_final) to exempt from the average. The dataset is then split into dataset 1
                                (t<t_init) and dataset 2 (t>t_final). The procedure is performed on dataset 1 and dataset 2
                                and their tranfsorms averaged again.
    :param output_on_full_harmonic_domain:
                                Whether the returned field is power distributed to the full harmonic domain.
    :return:
                                k_full, welch_averaged_ps and WINDOWS where windows contains elements such that
                                    first_window = WINDOWS[0]
                                    time_in_first_window, strain_in_first_window = first_window
    """
    print("Start: Calculating welch average")
    L_global = jnp.max(x)-jnp.min(x)
    if L_global < L:
        raise ValueError(f"Length of windows {L} larger than length of dataset {L_global}.")

    if leave_out is not None:
        x_init, x_final = leave_out

        cond_1 = jnp.where(x < x_init)
        cond_2 = jnp.where(x > x_final)
    else:
        cond_1 = jnp.where(x < jnp.inf)
        cond_2 = jnp.where(x > (-jnp.inf))

    y_strip_1 = y[cond_1]
    y_strip_2 = y[cond_2]

    x_in_strip_1 = x[cond_1]
    x_in_strip_2 = x[cond_2]

    if debug:
        welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2)

    check_even = (L_global % L == 0.)  # in general, for any integer L, this will not be true due to finite
    # sampling frequency! Will always be off by a bit.
    lf_edges_ds1, r_edges_ds1 = get_lr_edges(x_in_strip_1, y_strip_1, L, even=check_even)
    lf_edges_ds2, r_edges_ds2 = get_lr_edges(x_in_strip_2, y_strip_2, L, even=check_even)

    num = len(lf_edges_ds1) + len(lf_edges_ds2) if leave_out is not None else len(lf_edges_ds1)-1
    print(f"\nConstructing {num} windows over which we average.\n")

    if debug:
        welch_average_debug_plot_II(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2,
                                    lf_edges_ds1, r_edges_ds1, lf_edges_ds2, r_edges_ds2)

    collection_of_small_datasets_strain = []
    collection_of_small_datasets_times = []

    for left_lim, right_lim in zip(lf_edges_ds1, r_edges_ds1):
        idcs = jnp.where((x >= left_lim) & (x <= right_lim))
        collection_of_small_datasets_strain.append(y[idcs])
        collection_of_small_datasets_times.append(x[idcs])

    if leave_out is not None:
        for left_lim, right_lim in zip(lf_edges_ds2, r_edges_ds2):
            idcs = jnp.where((x >= left_lim) & (x <= right_lim))
            collection_of_small_datasets_strain.append(y[idcs])
            collection_of_small_datasets_times.append(x[idcs])
    else:
        # dataset 1 and dataset 2, so just append one of them.
        pass

    if debug:
        welch_average_debug_plot_III(collection_of_small_datasets_strain, collection_of_small_datasets_times, x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2)

    collection_of_small_datasets_strain_windowed = [d * tapering_function(d) for d in
                                                    collection_of_small_datasets_strain]

    if debug:
        welch_average_debug_plot_IV(collection_of_small_datasets_times, collection_of_small_datasets_strain,
                                    collection_of_small_datasets_strain_windowed)


    n_dtps = len(collection_of_small_datasets_times[0])
    dx = L/(n_dtps-1)

    F = lambda arr: fw_hartley(arr)

    data_fields = collection_of_small_datasets_strain_windowed

    empirical_power_spectra = jnp.array([power_analyze_re_hartley(y_values=h) for h in data_fields])
    k = jnp.fft.fftfreq(n=n_dtps, d=dx)
    ps = final_average_call(empirical_power_spectra, axis=0)

    S = collection_of_small_datasets_strain_windowed
    T = collection_of_small_datasets_times
    WINDOWS = jnp.array(list(zip(T,S)))

    if output_on_full_harmonic_domain:
        return k, ps, WINDOWS
    else:
        mask = (k > 0)
        return k[mask], ps[mask], WINDOWS


def get_lr_edges(x, y, L, even):
    """

    :param L:          Length of little windows
    :param x:          The global times.
    :param y:          The global strain.
    :param even:       Whether the total time is exactly divisible by L. If this is not the case, we need to discard
                       the very last window to upkeep equal-length windows. Visually clear when setting `debug` to True
                       in the `calculate_welch_average` function and inspecting the corresponding plots.
    :return:
    """
    lf_edges = jnp.arange(min(x), max(x), L)
    rght_edges = (lf_edges + L)
    if not even:
        rght_edges = rght_edges[:-1]
    return lf_edges, rght_edges


def welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2, show=True, alpha=1.):
    _ = plt.figure()
    plt.plot(x_in_strip_1, y_strip_1, label="Data in strip 1", alpha=alpha)
    plt.plot(x_in_strip_2, y_strip_2, label="Data in strip 2", alpha=alpha)
    usual_plot(title="Data (with a potential mask in a selected region)", show=show)


def welch_average_debug_plot_II(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2,
                                lf_edges_ds1, r_edges_ds1, lf_edges_ds2, r_edges_ds2):
    welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2, show=False, alpha=.2)

    ax = plt.gca()

    # Plot Rectangles to visualize
    heights = [4, 4.5, 5] * len(lf_edges_ds1)
    for i, (left, right) in enumerate(zip(lf_edges_ds1, r_edges_ds1)):
        width = right - left
        if i % 2 != 0:
            c = (1, 0, 0)
            lw = 1
            height = 4
        else:
            c = (0, 0, 1)
            lw = 2
            height = 5
        rect = Rectangle((left, 0), width, heights[i], facecolor='none', linewidth=lw, edgecolor=c)
        ax.add_patch(rect)

    # Plot Rectangles to visualize
    for i, (left, right) in enumerate(zip(lf_edges_ds2, r_edges_ds2)):
        width = right - left
        if i % 2 != 0:
            c = (1, 0, 0)
            lw = 1
            height = 4
        else:
            c = (0, 0, 1)
            lw = 2
            height = 5
        rect = Rectangle((left, 0), width, heights[i], facecolor='none', linewidth=lw, edgecolor=c)
        ax.add_patch(rect)

    plt.text(-2, -4,
             f"Everything not in these windows, I will throw away.\n Number of windows: {len(lf_edges_ds1) + len(lf_edges_ds2)}"
             f"if something cut out, else divide by 2.",
             fontsize=12)
    plt.xlabel("Time [s]", fontsize=12)
    plt.show()


def welch_average_debug_plot_III(collection_of_small_datasets_strain, collection_of_small_datasets_times, x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2):

    welch_average_debug_plot_I(x_in_strip_1, y_strip_1, x_in_strip_2, y_strip_2, show=False, alpha=.2)

    for d, t in zip(collection_of_small_datasets_strain, collection_of_small_datasets_times):
        plt.plot(t, d)

    usual_plot(title="Data windows to be used in the average; full dataset has alpha ~ 0.2")


def welch_average_debug_plot_IV(collection_of_small_datasets_times, collection_of_small_datasets_strain, collection_of_small_datasets_strain_windowed):
    fig5 = plt.figure(figsize=(10, 6))
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Strain $[10^{-19}]$")

    plt.plot(collection_of_small_datasets_times[0], collection_of_small_datasets_strain[0], label="Example of original data in a window")
    plt.plot(collection_of_small_datasets_times[0], collection_of_small_datasets_strain_windowed[0], label="The same but tapered")
    usual_plot()
