from dataclasses import dataclass
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gwosc.locate import get_event_urls
import matplotlib.pyplot as plt
import requests
import jax.numpy as jnp
from scipy.signal.windows import tukey
import numpy as np
import os
from typing import Literal, Optional, Any
import re

from .common_utils import unpickle_me_this
from ..basics.welch_average import calculate_welch_average

__all__ = ["get_sample_data", "iterative_midpoint_average", "power_analyze_re", "unpack_centered",
                 "convert_gps_to_seconds", "DEPR_get_strain_data", "get_time_and_strain_from_disc"]


def get_sample_data(norm=1e19, time_window=(15,17), end_points_small=False, taper=False):
    """
    Gets some exemplary data from first detected GravWave event.
    :param norm:                    Scaling the data for visual purposes.
    :param time_window:             Which data to pick out.
    :param end_points_small:        Whether the left and right bound should be chosen such that the first data point should
                                    be approximately 0.
    :param taper:                   Whether to apply a Tukey window.
    :return:
    """
    strain = unpickle_me_this(
        "/Users/iason/PycharmProjects/STRAIN/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle",
        absolute_path=True)

    zero_time = 1126259446  # I got this zero time by looking at the caption of the figure produced by strain.plot().
    time = np.array(strain.times) - zero_time  # in seconds

    full_data = norm * strain.value
    full_time = time.copy()

    t_min, t_max = time_window

    if end_points_small:
        eps1 = 0.1  # how small should the starting point be
        eps2 = 1e-6  # how near the endpoint be to the start point

        close_to_zero_idcs = np.where(np.abs(full_data) < eps1)
        close_to_zero_times = time[close_to_zero_idcs]
        t_min = np.max(close_to_zero_times[close_to_zero_times<t_min])

        d0 = full_data[np.where(time == t_min)]

        where_similar_to_d0 = np.where(np.isclose(full_data, d0, rtol=0, atol=0.001))

        time_similar_to_d0 = time[where_similar_to_d0]
        res_list = time_similar_to_d0 - t_max
        most_similar_to_t_max = np.min(np.abs(time_similar_to_d0 - t_max))

        t_max = most_similar_to_t_max + t_max

    indcs = np.where((t_min <= time) & (time <= t_max))
    data = full_data[indcs]
    time = time[indcs]

    if taper:
        tapering_function = lambda d: tukey(M=len(d), alpha=0.1, sym=True)
        data = tapering_function(data)*data

    return jnp.array(time), jnp.array(data)


def iterative_midpoint_average(data, n_iter=2, plot=False):
    """
    Gets the middle line (large scale structure) of the data and the standard deviation of the data about that
    middle line by iterative midpoint averaging, i.e. each iteration replaces x_i by
        (x_{i-1} + x_{i+1}) / 2.
    :param data:   array-like,  To analyze
    :param plot:   boolean,     If true, plots the data with the found middle line
    :return:
    """
    data = np.asarray(data, dtype=float)
    x = np.real(data).copy()

    for _ in range(n_iter):
        x_new = x.copy()
        x_new[1:-1] = 0.5 * (x[:-2] + x[2:])
        x = x_new

    middle = x
    residuals = np.real(data) - middle
    width = np.std(residuals)

    if plot:
        t = np.arange(len(x))
        plt.plot(t, np.real(data), alpha=0.5, label="data")
        plt.plot(t, middle, lw=2, label=f"middle ({n_iter} iters)")
        plt.fill_between(t, middle - width, middle + width, alpha=0.3)
        plt.legend()
        plt.show()

    print("Standard deviation of the data about middle line is: {:.3f} ".format(width), ", using iterative "
                                                                                        "midpoint averaging.")
    return middle, width



def power_analyze_re(x_values, y_values):
    """
    Returns an estimate of the power spectrum by absolute squaring the fourier transform.
    :param x_values: The x values used to determine the spacing needed to calculate the fourier modes.
    :param y_values: A real space periodic array.
    :return:
    """
    N = len(x_values)
    dx = x_values[1] - x_values[0]
    ps = jnp.abs(jnp.fft.fft(y_values, n=N, norm="ortho"))**2
    k = jnp.fft.fftfreq(N, d=dx)
    return k, ps


def unpack_centered_old(strain_series, gps_time, duration=32, center_at=None):
    times_gps = np.array(strain_series.times)
    strain = np.array(strain_series.value)

    center_gps = gps_time if center_at is None else center_at
    half = duration / 2

    start = center_gps - half
    end = center_gps + half

    mask = (times_gps >= start) & (times_gps <= end)

    # convert to seconds starting at (center_gps - half)
    masked_times_gps = times_gps[mask]
    times_sec = convert_gps_to_seconds(masked_times_gps, t0=start)

    strain_sec = jnp.array(strain[mask] * 1e19)
    times_sec = jnp.array(times_sec)

    return times_sec, strain_sec, start


def unpack_centered(strain_series, gps_time, duration=32, center_at=None):
    times_gps = np.array(strain_series.times)
    strain = np.array(strain_series.value)

    center_gps = gps_time if center_at is None else center_at
    half = duration / 2

    start = center_gps - half
    end = center_gps + half

    mask = (times_gps >= start) & (times_gps <= end)

    # convert to seconds starting at (center_gps - half)
    masked_times_gps = times_gps[mask]
    times_sec = convert_gps_to_seconds(masked_times_gps, t0=start)

    strain_sec = jnp.array(strain[mask] * 1e19)
    times_sec = jnp.array(times_sec)

    return times_sec, strain_sec, start


def convert_gps_to_seconds(gps_times, t0=None):
    gps_times = np.asarray(gps_times)
    if t0 is None:
        t0 = gps_times[0]
    times = gps_times - t0
    return times


def _get_strain_data(gps_center, absolute_path, desired_duration=32, unpack=True):
    """
    Get strain data based on HDF5 objects saved on the disk.

    :param gps_center:              A float around which the series is centered
    :param absolute_path:           The absolute path of the HDF5 file.
    :param unpack:                  If false, return the GWPY series object, else time and strain as tuple.
    :param desired_duration:        The returned time series will be centered on gps_center and start at
                                    gps_center - desired_duration/2 and end at gps_center + desired_duration/2.
    :return:
    """
    strain_series = TimeSeries.read(absolute_path, format='hdf5.gwosc')
    if not unpack:
        return strain_series
    time = np.array(strain_series.times) - gps_center
    strain = strain_series.value * 1e19

    left_time_bound = -desired_duration/2
    right_time_bound = +desired_duration/2

    to_keep = np.where((time > left_time_bound) & (time < right_time_bound))
    time_masked = time[to_keep]
    strain_masked = strain[to_keep]
    return time_masked, strain_masked



def DEPR_get_strain_data(gw_name = 'GW150914', detector = 'H1', duration = 32, unpack=True, center_at=None,
                    absolute_path=None, **kwargs):
    """
    Sebastian Gil's function, modified.

    This function takes the name of a confirmed gravitational
    wave transient and obtains the GPS time most closely matching
    the event. It then checks whether there exists a local copy of
    the published strain data corresponding to the event. If the data
    is not available, it downloads it before writing it out into a file.
    Following that, it turns the file into a timeseries object.

    This is fixed to the Hanford detector for now. Latter versions should
    account for Livingston and Virgo data as well.

    The dataset being downloaded is 32s long at a
    sampling rate of 4096 Hz.

    :param unpack: bool,    If true, return define t0 as 0 and return new times and strain scaled by 1e19.
    :param kwargs:          Keyword arguments to be passed to get_event_urls.

    """
    gps_time = center_at
    if absolute_path is None:
        # Fetch GPS time for the event
        gps_time = event_gps(gw_name)
        print("Fetching: ", gw_name)
        # Fetch event name from catalog
        url = get_event_urls('GW150914', detector='H1', format='hdf5', **kwargs)
        print(f"\tNo. of found urls: {len(url)}. Pinging urls[0]. List of all urls:\n\t", *url)
        url = url[0]
        # Choose a name for the file
        file_name = os.path.basename(url)
        # Check if the data has already been downloaded
        file_path = f'/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/gwpy_objects/{file_name}'
        if os.path.exists(file_path):
            print('The requested dataset exists locally.')
            strain_file = file_path
        else:
            print('The requested dataset does not exist locally. Downloading data for {}.'.format(file_name))
            with open(file_path,'wb') as strain_file:
                strain_data = requests.get(url)
                strain_file.write(strain_data.content)
        strain_series = TimeSeries.read(file_path, format='hdf5.gwosc')
    strain_series = TimeSeries.read(absolute_path, format='hdf5.gwosc')
    if not unpack:
        return strain_series
    return unpack_centered(strain_series, gps_time, duration=duration, center_at=center_at)


def whiten(y:np.array, amp:np.array, tapering_function = None):
    """

    :param y:                       The real-space data to whiten.
    :param amp:                     The custom amplitude spectrum to whiten with over the full harmonic domain.
    :param tapering_function:       The tapering function to use; default is a Tukey window.

    :return: The whitened data in real space. Normalization probably something like *N.
    """
    if tapering_function is None:
        tapering_function = lambda d: tukey(M=len(d), alpha=0.1, sym=True)

    y = y*tapering_function(y)
    y_harmonic = np.fft.fft(y)
    whitened_y_harmonic = y_harmonic / amp
    return np.fft.ifft(whitened_y_harmonic).real


@dataclass
class AuxData:
    ps_welch: Optional[Any] = None
    freqs: Optional[Any] = None
    win: Optional[Any] = None


@dataclass
class EventData:
    time: Any
    strain: Any
    gps: float
    gwpy: Any
    # optional whitening / Welch properties
    event_time: Optional[Any] = None
    event_strain: Optional[Any] = None
    event_strain_white: Optional[Any] = None
    aux: Optional[AuxData] = None


def _get_event_gps_from_readme(event_name, path_to_readme):
    with open(path_to_readme, "r") as fn:
        content = fn.read()

    # Split into event blocks (each starts with H1_... and L1_...)
    blocks = re.split(r"\n\s*\n", content)

    matching_blocks = [b for b in blocks if event_name in b]

    if len(matching_blocks) == 0:
        raise ValueError(f"Expected at least one block for {event_name} in readme file, but found None")

    block = matching_blocks[0]

    # Extract GPS value
    match = re.search(r"GPS:\s*([0-9.]+)", block)
    if match is None:
        raise ValueError(f"No GPS entry found for {event_name}")

    return float(match.group(1))


def _get_window_containing_zero(WINDOWS):
    matches = []

    for event_time, strain_time in WINDOWS:
        if (event_time[0] <= 0.0) and (0.0 <= event_time[-1]):
            matches.append((event_time, strain_time))

    if len(matches) != 1:
        raise ValueError(f"Expected exactly one window containing 0., found {len(matches)}")

    return matches[0]  # (event_time, strain_time)


def get_time_and_strain_from_disc(event_name="GW150914", detector:Literal["H1", "L1"]= "H1",
                                  data_duration:Literal["32sec", "4096sec"]="4096sec",
                                  desired_duration=32, add_whitened_data=False, **kwargs):
    """
    Given the unique name of an event like GW150914, retrieves time and strain values as well as the corresponding
    gwpy TimeSeries class. All of these are stored as properties of a dict:

    GW150914 =  get_time_and_strain_from_disc()

        GW150914.time           The time in seconds centered around the event.
        GW150914.strain         The strain of the event in units of 1e-19.
        GW150914.gps            The GPS time of the event in seconds.
        GW150914.gwpy           The original gwpy TimeSeries object from which these values were extracted.

        if add_whitened_data, the data is subdivided into windows from which the Welch average is calculated.
        Following properties are then additionally provided:

        GW150914.event_time             The times of the welch average window containing the event
        GW150914.event_strain           The strain of the welch average window containing the event
        GW150914.event_strain_white     The whitened strain of the welch average window containing the event

        GW150914.aux            A class containing the following attributes:
                                    GW150914.aux.ps_welch   :  The welch average on the full harmonic domain
                                    GW150914.aux.freqs      :  Corresponding frequencies
                                    GW150914.aux.win        :  All windows used in the Welch average. E.g.:
                                                                first_window = WINDOWS[0]
                                                                time_in_first_window, strain_in_first_window
                                                                = first_window


    For this workflow to work, a data folder containing hdf5 files must exist, along with a readme file containing metadata of every event
    stored in that folder.

    :type event_name:           A string of the unique name/ID of the GW event.
    :type desired_duration:     Will return a time series of approximately this duration.
    :type data_duration:        Actual duration of data. Either 32s or 4096s
    :type detector:             Whether to grab L1 or H1 data.
    :type add_whitened_data:    If True, add whitened data.
    :type kwargs:               If add_whitened_data is True, keywords arguments passed to `calculate_welch_average`.
    :return:
    """
    base_path = "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/gwpy_objects_II"

    hdf5_files = []
    readme = None
    for file in os.listdir(base_path):
        if file.endswith(".hdf5"):
            hdf5_files.append(file)
        if file.endswith(".txt"):
            readme = file
        if not file.endswith(".txt") and not file.endswith(".hdf5"):
            print("Unknown file ending for ", file, "... ignoring.")

    if readme is None:
        raise ValueError("Readme file not found.")

    matches = [fn for fn in hdf5_files if detector in fn and event_name in fn and data_duration in fn]

    if len(matches) != 1:
        raise ValueError(f"Expected exactly one match, found {len(matches)}: {matches}")

    match = matches[0]
    gps_time = _get_event_gps_from_readme(event_name, path_to_readme=os.path.join(base_path, readme))

    kwargs_unpack = {'desired_duration': desired_duration, 'unpack': True, 'gps_center': gps_time,
     'absolute_path': os.path.join(base_path, match)}

    kwargs_gwpy_object = kwargs_unpack.copy()
    kwargs_gwpy_object["unpack"] = False

    times, strain = _get_strain_data(**kwargs_unpack)
    gwpy_object = _get_strain_data(**kwargs_gwpy_object)

    obj = EventData(time=times, strain=strain, gps=gps_time, gwpy=gwpy_object)

    if add_whitened_data:
        freqs, ps, windows = calculate_welch_average(
            x=times, y=strain, output_on_full_harmonic_domain=True, **kwargs
        )

        # get N from first window
        N = len(windows[0][0])  # windows[0] = (time_window, strain_window)

        # find index of zero in the full times array
        zero_idx = (np.abs(times)).argmin()

        # Now just cut N//2 to the left and to the right to get the arrays containing the event. Note that the
        # Welch average will by construction cut right through the signal, since we're centered at the signal and
        # use a window length of L=2 (by default).
        half_N = N // 2
        start_idx = max(zero_idx - half_N, 0)
        end_idx = start_idx + N
        if end_idx > len(times):
            end_idx = len(times)
            start_idx = end_idx - N  # ensure length N

        # slice the arrays
        event_time = times[start_idx:end_idx]
        event_strain = strain[start_idx:end_idx]

        # event_time, event_strain = _get_window_containing_zero(windows)

        event_strain_whitened = whiten(y=event_strain, amp=jnp.sqrt(ps))

        obj.event_time = event_time
        obj.event_strain = event_strain
        obj.event_strain_white = event_strain_whitened
        obj.aux = AuxData(ps_welch=ps, freqs=freqs, win=windows)

    return obj
