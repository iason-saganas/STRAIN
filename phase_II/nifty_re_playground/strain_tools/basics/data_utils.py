from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps
from gwosc.locate import get_event_urls
import requests
import os
import jax.numpy as jnp
from scipy.signal.windows import tukey

from .common_utils import unpickle_me_this

__all__ = ["get_sample_data", "iterative_midpoint_average", "power_analyze_re", "unpack_centered",
                 "convert_gps_to_seconds", "get_strain_data"]

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
        "/phase_I/partial_successful_reconstruct_and_where_is_the_signal/store/GW150914_strain.pickle",
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
    return gps_times - t0


def get_strain_data(gw_name = 'GW150914', detector = 'H1', duration = 32, unpack=True, center_at=None,
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
    print("hey=? ")
    return unpack_centered(strain_series, gps_time, duration=32, center_at=center_at)