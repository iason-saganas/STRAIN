# Updated version of `phase_II/nifty_re_playground/_00_gw_search.py`
import os
from typing import Literal, Any, Optional
from dataclasses import dataclass

from phase_II.nifty_re_playground.strain_tools import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view  # Greetings to Claude LLM for the tipp

"""
Note 1: This function is something that should be run in parallel on a cluster; instead of window by window 
        search, the stress can be calculated for all windows in parallel.
        
Note 2: It could be that a signal resides exactly on a window edge, which is why we should actually take 50% overlapping
        windows in the Welch-average so that in the Stress calculation, the chirp isn't broken  
    
Given an event name and other parameters, performs a search as follows. 

:param stationarity_time_scale: int,        A.k.a.: `T_global_welch` 
                                            The time in seconds over which the data is assumed to be stationary, 
                                            i.e. has negligible drift, such that it is safe to compute the 
                                            Welch average over this time. Most important parameter, since influences
                                            actual data duration and T_mini_welch.
:param desired_max_data_duration, int       The upper bound on much data to analyze. Will be cropped such that it is 
                                            the largest multiple of T_global_welch.

The retrieved data will be divided into `num_segments`-many, `T_global_welch`-long segments. 

:param T_mini_welch: int,                   For each segment over which the Welch average is computed, determines 
                                            the length of the windows used to subdivide the data again for the 
                                            averaging procedure.   

We then have `num_segments`-many noise power spectra. For each of the segments' windows, the Stress is computed
using the current Welch-average estimate and the detection statistic is formed. Finally, a detection statistic time 
series is 'stitched together' from the results of all windows.

To summarize:
The data is divided into segments over which the Welch average is computed. Each segment is divided into 
windows needed for the Welch-average; the detection statistic is computed for each window and the results of all 
windows are stitched together. 
"""



class GwSearch:
    def __init__(self, event:str, detector:Literal["H1", "L1"], data_duration:Literal["32sec", "4096sec"],
                 desired_maximal_data_duration:float, random_center:float, stationarity_time_scale=32, T_mini_welch=2,
                 signal_duration_overlap=.3, debug_plots=False, debug_welch_average_plots=False,
                 out_name=""):
        """

        Tests for the presence of GWs in a data strip by computing and manipulating the Wigner function over multiple
        segments. Because a signal may land exactly on the edges of the segments, we overlap the segments.

        :param event:                           The unique event ID, e.g. 'GW150914'
        :param detector:                        Either 'L1' or 'H1'.
        :param data_duration:                   The duration of the strain data file sitting on the disk. Used to search
                                                for the data.
        :param desired_maximal_data_duration:   How much data to actually use at most. Will potentialy cropped to make
                                                multiple of `stationarity_time_scale`.
        :param random_center:                   Gps time on which to center the data, e.g. 5s before event:
                                                1126259462.4-5.
        :param stationarity_time_scale:         The timescale in seconds over which the data is assumed to be
                                                stationary.
        :param T_mini_welch:                    The length of the windows used to subdivide the data for the
                                                computation of the Welch-average.
        :param signal_duration_overlap:         In seconds, how much to overlap each segment with its left neighbor.
                                                Half of this value is cropped to the left and to the right of the
                                                detection statistic.
        :param out_name:                        Where to store intermediate results
        :param debug_plots:
        :param debug_welch_average_plots:
        """
        os.makedirs(out_name, exist_ok=True)

        # Let's start from the known event gps time and off-center the data a bit to simulate a random data pick
        random_center = 1126259462.4-5  # Beginning of GW150914 - 11 seconds

        T_global_welch = stationarity_time_scale
        # Add trimming segments and make integer multiple, get data
        num_segments = desired_maximal_data_duration // T_global_welch
        data_duration_modified = num_segments * T_global_welch  # I add another segment here to 'trim off' later
        obj = get_strain_from_disc(event_name=event, detector=detector, data_duration=data_duration,
                                   desired_duration=data_duration_modified, center_on_event=False, custom_center=random_center)
        global_time = obj.time
        global_strain = obj.strain - np.mean(obj.strain)  # detrend
        t_init = global_time[0]
        T_global = np.max(global_time)-np.min(global_time)

        print("User requested")
        print(f"{'\tMax. data duration [s]:':45}{desired_maximal_data_duration:>5}")
        print(f"{'\tAssumed stationarity time-scale [s]:':45}{T_global_welch:>5}")
        print(f"{'\tWelch-average length [s]:':45}{T_mini_welch:>5}")
        print("Query for ", data_duration_modified, " to make duration multiple of stationarity time-scale...")
        print("Got ", T_global, " seconds of data.\n")

        if T_global % T_global_welch != 0.:
            raise ValueError("Please ensure that the gotten time series duration is a multiple of the stationarity time-scale."
                             "for simplicity.")
        if not isinstance(T_mini_welch, int):
            raise ValueError("There is a very sneaky shape bug when setting T_mini_welch to a float. Fix in the future.")

        if debug_plots:
            plt.title("Data for event " + event)
            plt.vlines(obj.gps_event-obj.gps_center, np.min(global_strain), np.max(global_strain), label="Event", lw=2, color=red)
            plt.plot(global_time, global_strain)
            plt.xlabel("Time (s)")
            plt.ylabel("Strain")
            plt.legend()
            plt.show()

        segment_times = []
        segment_strains = []

        uneven_welch_average = not (T_global_welch % T_mini_welch == 0)
        # If this quantity is uneven, the Welch-average actually calculates a segment length as
        # T_global_welch - (T_global_welch % T_mini_welch), i.e. the last window is thrown out because it is not of
        # length T_mini_welch. This has to be accounted for.

        rest = T_global_welch % T_mini_welch  # 0 in the best case

        segment_duration = T_global_welch - rest
        for segment_idx in range(num_segments):

            t_end = t_init + segment_duration  # each segment potentially shortened

            slice_array = np.where( (global_time >= t_init) & (global_time <= t_end))

            segment_time = global_time[slice_array]
            segment_strain = global_strain[slice_array]

            segment_times.append(segment_time)
            segment_strains.append(segment_strain)

            t_init = t_end  # next segment starts immediately after the previous

        if debug_plots:
            print("Subdividing the data of length ", data_duration_modified, " into ", num_segments, " segments.")
            print("This plot shows those segments stitched together and should look like the normal data array.")

            plt.title("Data segments for event " + event)
            for t, s in zip(segment_times, segment_strains):
                plt.plot(t, s)
            plt.xlabel("Time (s)")
            plt.ylabel("Strain")
            plt.show()

        T_global = segment_times[-1][-1]
        if uneven_welch_average:
            print("Note: the Welch-average length does not cleanly divide the stationarity time-scale. Therefore, clipping"
                  "data to ", T_global, " seconds.")

        # Take each segment and calculate the Welch average
        freqs = None
        welch_averages = []
        windows_of_all_segments = []  # elements (segment_idx, time_window, strain_window) for ALL segments
        for segment_idx, (t,s) in enumerate(zip(segment_times, segment_strains)):
            freqs, welch_ps, windows = calculate_welch_average(x=t, y=s, L=T_mini_welch, output_on_full_harmonic_domain=True,
                                                               debug=debug_welch_average_plots)

            welch_ps = welch_ps[:-1]  # seems like the absolute value maximum is repeated twice, so f is
            # [0, +f_nyq, ..., -f_nyq] instead of [0, +f_nyq, ..., -f_nyq)
            # windows is a list, such that:
            # first_window = windows[0]
            # time_in_first_window, strain_in_first_window = first_window
            insert_segment_idx_into_window = [(segment_idx, tp[0], tp[1]) for tp in windows]
            windows_of_all_segments.extend(insert_segment_idx_into_window)
            welch_averages.append(welch_ps)
            print(f"Welch average {segment_idx+1}/{len(segment_times)}: Done")


        if debug_plots:
            plt.title("All windows used for Welch-averaging")
            for (segment_idx, t, s) in windows_of_all_segments:
                plt.plot(t, s)

                middle_time = np.mean(t)
                min_strain = np.min(s)
                plt.text(middle_time, min_strain, s=str(segment_idx+1))
            plt.xlabel("Time (s)")
            plt.ylabel("Strain")
            plt.show()

        # Base fields
        self.out_name = out_name
        self.detector = detector
        self.global_time = global_time
        self.global_strain = global_strain
        self.T_mini_welch = T_mini_welch
        self.signal_duration_overlap = signal_duration_overlap
        self.segment_times = segment_times
        self.num_segments = num_segments
        self.debug_plots = debug_plots
        self.welch_averages = welch_averages
        self.freqs = freqs[:-1]  # my welch average method includes both positive and negative nyquist.

        # Derived fields or to be set
        self.fs = 1 / (self.global_time[1] - self.global_time[0])
        self.untrimmed_detection_stats = None
        self.trimmed_detection_stats = None


    def get_overlapping_window_segments(self):

        n_samples_window = int(self.T_mini_welch * self.fs)
        n_samples_step = max(1, int((self.T_mini_welch - self.signal_duration_overlap) * self.fs))

        t_windows_full = sliding_window_view(self.global_time, n_samples_window)[::n_samples_step]
        s_windows_full = sliding_window_view(self.global_strain, n_samples_window)[::n_samples_step]

        t_windows = [tw for tw in t_windows_full if len(tw) == n_samples_window]  # throw away possibly bad window at the end
        s_windows = [sw for sw in s_windows_full if len(sw) == n_samples_window]

        windows_of_all_overlapping_segments = [
            (idx, t_win, s_win)  # -1 or None for segment index, since we're ignoring segments now
            for idx, (t_win, s_win) in enumerate(zip(t_windows, s_windows))
        ]

        if self.debug_plots:

            for segment_idx, t_win, s_win in windows_of_all_overlapping_segments:
                plt.plot(t_win, s_win, lw=0.8, alpha=0.5)
                welch_ps_idx_plot = self._get_welch_index(t_win, s_win) + 1
                plt.text(np.mean(t_win), np.min(s_win), str(welch_ps_idx_plot),
                         fontsize=7, ha='center', va='bottom', color='k')

            plt.xlabel("Time (s)")
            plt.ylabel("Strain")
            plt.title("Overlapping windows for Wigner analysis (color = segment)")
            plt.show()

        return windows_of_all_overlapping_segments


    def _get_welch_index(self, t_seg, s_seg):
        segment_edges = [t[0] for t in self.segment_times] + [self.segment_times[-1][-1]]
        middle_time = np.mean(t_seg)
        min_strain = np.min(s_seg)

        # assign Welch PSD by the segment containing the window center
        segment_idx = np.searchsorted(segment_edges, middle_time, side='right') - 1
        segment_idx = min(segment_idx, self.num_segments - 1)  # just in case
        return segment_idx


    def sliding_wigner_function_search(self):

        files = os.listdir(self.out_name)

        file_exists = False
        fn = None
        for file in files:
            if f"{self.detector}_untrimmed_detection_stats_T_{self.T_mini_welch}sec" in file:
                fn = file
                file_exists = True
                break

        if file_exists:
            untrimmed_detection_stats = unpickle_me_this(self.out_name + "/" + fn)
        else:
            print("Calculating stress and detection statistics for each window of each segment with overlap.")
            windows_of_all_overlapping_segments = self.get_overlapping_window_segments()
            untrimmed_detection_stats = []
            for window_idx, (segment_idx, t, s) in enumerate(windows_of_all_overlapping_segments):
                # windows_of_all_segments elements : (segment_idx, time_window, strain_window) over all windows and segments
                welch_ps_idx = self._get_welch_index(t, s)
                welch_to_use = self.welch_averages[welch_ps_idx]

                s = s - np.mean(s)

                amp = np.sqrt(welch_to_use)
                whitened_data = whiten(y=s, amp=amp)
                stress, _, _ = Stress_jft(xi=whitened_data, time=t, supress_print=True)
                print("Stress calculation done for window ", window_idx+1 ,f"/ {len(windows_of_all_overlapping_segments)}")
                dc_line, _ = detection_statistic(stress_matrix=stress, plot=False)

                untrimmed_detection_stats.append((t,dc_line))
            name = self.out_name + f"/{self.detector}_untrimmed_detection_stats_T_{self.T_mini_welch}sec"
            pickle_me_this(name, untrimmed_detection_stats)

        print("Saving ", len(untrimmed_detection_stats), " overlapping windows.")
        self.untrimmed_detection_stats = untrimmed_detection_stats

        # Remove edges.
        trimmed_detection_stats = []
        n_samples_edges = int(self.fs * self.signal_duration_overlap) // 2  # assume even. If not error by one or
        # two samples
        for times, dc_lines in self.untrimmed_detection_stats:
            if n_samples_edges == 0:
                # Overlap was 0 => Don't cut
                t = times
                dc_line = dc_lines
            else:
                # Non-zero overlap => Cut away half of the assumed signal duration to the left and the right
                # This way, one segement ends signal_duration_overlap/2 earlier, whereas the next segment starts
                # signal_duration_overlap/2 later, so they form a boundary
                t = times[n_samples_edges:-n_samples_edges]
                dc_line = dc_lines[n_samples_edges:-n_samples_edges]
            trimmed_detection_stats.append((t, dc_line))

        self.trimmed_detection_stats = trimmed_detection_stats


    def plot_detection_statistic(self, ax=None, mode="longer", **kwargs):
        if self.untrimmed_detection_stats is None:
            raise ValueError("No untrimmed detection statistics available; run `sliding_wigner_function_search` first")

        detection_times = np.concatenate([el[0] for el in self.trimmed_detection_stats])
        detection_values = np.concatenate([el[1] for el in self.trimmed_detection_stats])

        if ax is None:
            _ = plt.figure(figsize=(8, 4))
            ax = plt.gca()

        ax.plot(detection_times, detection_values/1e3, color="black")

        ax.text(
            0.05, 0.9,  # x, y in axes fraction (0–1)
            self.detector,  # text
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=20,
        )

        thesis_plot(yl=r"$\mathrm{DS}(t)\cdot 10^{-3}$", mode=mode, custom_ax=ax,**kwargs)
        return ax


    def plot_segments(self):
        windows_of_all_overlapping_segments = self.get_overlapping_window_segments()
        for segment_idx, t_win, s_win in windows_of_all_overlapping_segments:
            plt.plot(t_win, s_win, lw=0.8, alpha=0.5)
            welch_ps_idx_plot = self._get_welch_index(t_win, s_win) + 1
            plt.text(np.mean(t_win), np.min(s_win), 'ps'+str(welch_ps_idx_plot),
                     fontsize=7, ha='center', va='bottom', color='k')
            plt.text(np.mean(t_win), np.max(s_win), 'seg No. ' + str(segment_idx),
                     fontsize=7, ha='center', va='bottom', color='k')

        plt.xlabel("Time (s)")
        plt.ylabel("Strain")
        plt.title("Overlapping windows for Wigner analysis (color = segment)")
        plt.show()


    def plot_wigner_for_segment(self, segment_idx, return_segment_data=False, plot=True, padding=0.):
        windows_of_all_overlapping_segments = self.get_overlapping_window_segments()
        _, t, s = [seg for seg in windows_of_all_overlapping_segments if seg[0]==segment_idx][0]

        s = s-np.mean(s)

        welch_ps_idx = self._get_welch_index(t, s)
        welch_to_use = self.welch_averages[welch_ps_idx]
        amp = np.sqrt(welch_to_use)
        whitened_data = whiten(y=s, amp=amp)
        print("Calculating stress and associated quantities for segment with index No. ", segment_idx)
        stress, dual_time, dual_freq = Stress_jft(xi=whitened_data, time=t, supress_print=True)
        # stress, dual_time, dual_freq = Stress_jft_experimental(xi=whitened_data, time=t, supress_print=True,
        #                                                        tukey_window_where_necessary=True,
        #                                                        padding_extent=padding)

        dc_line, _ = detection_statistic(stress_matrix=stress, plot=False)

        stress_smoothed = smooth_matrix(stress, smoothing_lvl=5)
        SWP = stress_smoothed.real**2

        if plot:
            fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
            axs[0][1].sharex(axs[0][0])
            axs[1][0].sharex(axs[0][0])
            axs[0][1].sharey(axs[0][0])
            axs[1][0].sharey(axs[0][0])

            vis_stress_kwargs=dict(delay_plot=True, rows=dual_freq, cols=dual_time, colorbar_label="")
            visualize_stress(stress_matrix=stress, custom_ax=axs[0][0], **vis_stress_kwargs)
            visualize_stress(stress_matrix=SWP, custom_ax=axs[0][1], **vis_stress_kwargs)
            visualize_stress(stress_matrix=stress_smoothed, custom_ax=axs[1][0], **vis_stress_kwargs)

            axs[1][1].plot(dual_time, dc_line)
            axs[1][1].vlines([t[0]+self.signal_duration_overlap,t[-1]-self.signal_duration_overlap],ymin=0,
                             ymax=dc_line.max(), label="Overlap boundaries", color=red)

            axs[0][0].set_ylabel(r"Frequency $f$ $\mathrm{[Hz]}$")
            axs[1][0].set_ylabel(r"Frequency $f$ $\mathrm{[Hz]}$")
            axs[1][1].set_ylabel(r"Amplitude")

            axs[1][0].set_xlabel(r"Time $t$ $\mathrm{[s]}$")
            axs[1][1].set_xlabel(r"Time $t$ $\mathrm{[s]}$")

            axs[1][1].legend()

            save_figure(save_fig=False, show=True, tight_ly=False)

        if return_segment_data:
            return SegmentData(wigner=stress, wigner_smoothed=stress_smoothed, wigner_smoothed_squared=SWP,
                               time=t, strain=s, dc_line=dc_line)


    def plot_noise_power_spectrum(self, segment_idx):
        windows_of_all_overlapping_segments = self.get_overlapping_window_segments()
        _, t, s = [seg for seg in windows_of_all_overlapping_segments if seg[0] == segment_idx][0]
        welch_ps_idx = self._get_welch_index(t, s)
        welch_to_use = self.welch_averages[welch_ps_idx]

        _ = plt.figure(figsize=(8.,4.))
        plt.plot(self.freqs, welch_to_use)
        plt.loglog()
        thesis_plot(mode="longer", xl="Frequency", yl="Power")


@dataclass
class SegmentData:
    wigner: Any
    wigner_smoothed: Any
    wigner_smoothed_squared: Any
    time: np.array
    strain: np.array
    dc_line: Optional[Any] = None