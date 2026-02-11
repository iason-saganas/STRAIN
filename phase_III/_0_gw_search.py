# Updated version of `phase_II/nifty_re_playground/_00_gw_search.py`
from phase_II.nifty_re_playground.strain_tools import *
import numpy as np
import matplotlib.pyplot as plt

"""
Note 1: This function is something that should be run in parallel on a cluster; instead of window by window 
        search, the stress can be calculated for all windows in parallel.
        
Note 2: It could be that a signal resides exactly on a window edge, which is why we should actually take 50% overlapping
        windows in the Welch-average.      
    
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

# Parameters
event = "GW150914"
detector = "L1"
data_duration = "4096sec"
debug_plots = False
debug_welch_average_plots = False
stationarity_time_scale = 32
T_mini_welch = 3
desired_maximal_data_duration = 80

T_global_welch = stationarity_time_scale
# Add trimming segments and make integer multiple, get data
num_segments = desired_maximal_data_duration // T_global_welch
data_duration_modified = num_segments * T_global_welch  # I add another segment here to 'trim off' later
obj = get_time_and_strain_from_disc(event_name=event, detector=detector, data_duration=data_duration,
                                    desired_duration=data_duration_modified, center_on_event=False)
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

if t_init != 0.:
    raise ValueError("Please ensure that the time series starts at t=0. by construction")
if T_global % T_global_welch != 0.:
    raise ValueError("Please ensure that the gotten time series duration is a multiple of the stationarity time-scale."
                     "for simplicity.")
if not isinstance(T_mini_welch, int):
    raise ValueError("There is a very sneaky shape bug when setting T_mini_welch to a float. Fix in the future.")

if debug_plots:
    plt.title("Data for event " + event)
    plt.plot(global_time, global_strain)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.show()

segment_times = []
segment_strains = []

uneven_welch_average = not (T_global_welch % T_mini_welch == 0)
# If this quantity is uneven, the Welch-average actually calculates a segment length as
# T_global_welch - (T_global_welch % T_mini_welch), i.e. the last window is thrown out because it is not of
# length T_mini_welch. This has to be accounted for.

rest = T_global_welch % T_mini_welch  # 0 in the best case

t_init = 0
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
welch_averages = []
windows_of_all_segments = []  # elements (segment_idx, time_window, strain_window) for ALL segments
for segment_idx, (t,s) in enumerate(zip(segment_times, segment_strains)):
    freqs, welch_ps, windows = calculate_welch_average(x=t, y=s, L=T_mini_welch, output_on_full_harmonic_domain=True,
                                                       debug=debug_welch_average_plots)
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

print("Calculating stress and detection statistics for each window of each segment.")

detection_stats = []
for window_idx, (segment_idx, t, s) in enumerate(windows_of_all_segments):
    # windows_of_all_segments elements : (segment_idx, time_window, strain_window) over all windows and segments
    welch_to_use = welch_averages[segment_idx]
    amp = np.sqrt(welch_to_use)
    whitened_data = whiten(y=s, amp=amp)
    stress, _, _ = Stress_jft(xi=whitened_data, time=t, supress_print=False)
    print("Stress calculation done for window ", window_idx+1 ,f"/{len(windows_of_all_segments)}")
    dc_line, _ = detection_statistic(stress_matrix=stress, plot=False)
    detection_stats.append((t,dc_line))

detection_times = np.concatenate([el[0] for el in detection_stats])
detection_values = np.concatenate([el[1] for el in detection_stats])
plt.plot(detection_times, detection_values)
plt.savefig("test 2")