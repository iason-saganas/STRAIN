
from phase_III.gw_search_module import *
from phase_III.strain import *
import matplotlib.pyplot as plt
import numpy as np

searching_kwargs = dict(event="GW150914", data_duration="4096sec", desired_maximal_data_duration=32,
                    random_center=1126259462.4-5, stationarity_time_scale=32, T_mini_welch=2, signal_duration_overlap=0.6,
                        debug_plots=False, debug_welch_average_plots=False, out_name="gw_search")

H1_searcher = GwSearch(detector="H1", **searching_kwargs)
L1_searcher = GwSearch(detector="L1", **searching_kwargs)

H1_searcher.sliding_wigner_function_search()
L1_searcher.sliding_wigner_function_search()


# H1_searcher.plot_segments()
# H1_searcher.plot_detection_statistic()
# H1_searcher.plot_wigner_for_segment(segment_idx=14)
# H1_searcher.plot_wigner_for_segment(segment_idx=8)

fig, axs = plt.subplots(3, 1, figsize=(8., 8.))
axs[0].sharex(axs[1])
axs[1].sharex(axs[2])
axs[0].tick_params(labelbottom=False)
axs[1].tick_params(labelbottom=False)

H1_searcher.plot_detection_statistic(show=False, ax=axs[0], xl="", mode="square")
L1_searcher.plot_detection_statistic(show=False, ax=axs[1], xl="", mode="square")


windows = [
    (t, dc_H1, dc_L1)
    for (t, dc_H1), (_, dc_L1) in zip(L1_searcher.trimmed_detection_stats,
                                      H1_searcher.trimmed_detection_stats)
]


def cross_correlate(d_1:np.array, d_2:np.array, t_1:np.array, t_2:np.array, use_mathematically_correct_normalization=False):
    if not np.all(t_1 == t_2):
        raise ValueError("The two time arrays must be the same.")
    t = t_1

    d_1 = d_1 - np.mean(d_1)
    d_2 = d_2 - np.mean(d_2)

    assert len(d_1) == len(d_2), "The two data arrays must have the same length."

    n = len(d_1)
    l = np.linspace(-(n-1), (n-1), 2*(n-1)+1)
    f = 1/(t[1]-t[0])
    taus = l/f

    # loop for all l's:

    correlations = []
    for l_star in l:
        l_star = int(l_star)
        # loop for a single l_star:
        summands = []
        for i in range(int(n-np.abs(l_star))):  # upper limit: n-l-1
            summands.append(d_1[i] * d_2[i+l_star])

        if use_mathematically_correct_normalization:
            normalization = n-np.abs(l_star)  # mathematically correct
        else:
            normalization = n  # suppresses regions of large delays, which we are not interested in because there can't be physically sourced correlations
            # after dividing with the noise power spectrum estimate that are on the order of ~10ms! These would be random noise correlations ... I think
        res_for_l_star = sum(summands) / normalization
        correlations.append(res_for_l_star)

    correlations = np.array(correlations)/(np.std(d_1)*np.std(d_2))   # normalize with the standard deviations if you want
    return taus, correlations  # return time delays and C(τ)


def cross_correlate_fast(d_1: np.array, d_2: np.array, t_1: np.array, t_2: np.array,
                         ligo_normalization=True):
    if not np.all(t_1 == t_2):
        raise ValueError("The two time arrays must be the same.")

    d_1 = d_1 - np.mean(d_1)
    d_2 = d_2 - np.mean(d_2)

    n = len(d_1)
    taus = np.arange(-n + 1, n) * (t_1[1] - t_1[0])

    if not ligo_normalization:
        # mathematically correct normalization per lag
        norm = np.array([n - abs(l) for l in range(-n + 1, n)])
        corr = np.correlate(d_1, d_2, mode='full') / norm
    else:
        # simple normalization
        corr = np.correlate(d_1, d_2, mode='full') / n

    corr /= (np.std(d_1) * np.std(d_2))

    return taus, corr

all_correlation_tuples = []
for idx, (t, y1, y2) in enumerate(windows):
    print("cross correlating the pairs of idx", idx+1, " / ", len(windows))
    # tmp = cross_correlate(d_1=y1, d_2=y2, t_1=t, t_2=t)
    delays, corrs = cross_correlate_fast(d_1=y1, d_2=y2, t_1=t, t_2=t, ligo_normalization=True)
    # tmp = xcorr_data(strain_windows_h1[idx], strain_windows_l1[idx])
    all_correlation_tuples.append((delays, corrs))

max_correlation_in_each_window = [corr[np.argmax(np.abs(corr))] for tau, corr in all_correlation_tuples]
argmax_delay_in_each_window = [tau[np.argmax(corr)] for tau, corr in all_correlation_tuples]
time_windows = [t for t, _, _ in windows]
time_middle_points = [np.mean(t) for t in time_windows]
time_edges = [t[0] for t in time_windows]

ax = axs[2]

ax.plot(time_middle_points, max_correlation_in_each_window, marker=".", color="black", markersize=10)

light_travel_time = 10e-3
candidates_times = []
candidates_max_correlations = []
for idx, tau_argmax in enumerate(argmax_delay_in_each_window):
    if np.abs(tau_argmax) < light_travel_time:
        candidates_times.append(time_middle_points[idx])
        candidates_max_correlations.append(max_correlation_in_each_window[idx])
ax.plot(candidates_times, candidates_max_correlations, marker="o", markeredgecolor=blue,
        markersize=15, lw=0, markerfacecolor=(0,0,0,0), label=r"Delay $\vert\tau\vert$ under max. light travel time",
        color=(0,0,0,0))

# Plot the segment edges
for edge in time_edges:
    ax.axvline(edge, color='gray', linestyle='--', alpha=1)

ax.set_xlabel(r"Time $t$ $\mathrm{[s]}$")
ax.set_ylabel(r"$\mathrm{max}(C(\tau))$")
ax.legend(frameon=True, loc="upper left")
save_figure(show=True, save_fig=False, tight_ly=True)