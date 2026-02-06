from phase_II.nifty_re_playground.strain_tools import *
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from phase_II.utils.helpers import whiten, bandpass
import jax
jax.config.update("jax_enable_x64", True)

save_results_path = "/Users/iason/PycharmProjects/STRAIN/phase_II/gw_search_results/"
off_center = 1.5
events = [
    {"gw_name": 'GW150914', "duration":32, "unpack":True, "version": 3, "sample_rate":4096, "center_at": 1126259462.4+off_center},  # GW150914 source
    {"gw_name": 'GW150914', "duration":4096, "unpack":True, "version": 4, "sample_rate":4096, "center_at": 1126259598-4+off_center},  # GW150914 Glitch
    {"gw_name": 'GW250114_082203', "duration":32, "unpack":True, "version": 2, "sample_rate":4, "center_at": 1420878141.2+off_center},  # GW250114_082203 source
    {"gw_name": 'GW190521_074359', "duration":32, "unpack":True, "version": 2, "sample_rate":4, "center_at": 1242459857.4+off_center},  # GW190521_074359 source
]
event_idx = 3
event = events[event_idx]
name_of_event = event["gw_name"]

# abs_path_to_use = None
abs_path_to_use = "/Users/iason/PycharmProjects/STRAIN/data/data_pickle_or_hdf5/gwpy_objects/H-H1_GWOSC_4KHZ_R1-1242459842-32.hdf5"

times, strain, t0_gps = get_strain_data(**event, absolute_path=abs_path_to_use)
# strain_object = get_strain_data(**event, absolute_path=abs_path_to_use)
# plt.plot(strain_object.times, strain_object.whiten().bandpass(30, 300))
# plt.plot(strain_object.times, strain_object.value)
# usual_plot()
# stop

marker = convert_gps_to_seconds(event["center_at"]-off_center, t0=t0_gps)
# plt.vlines(marker, -5, 5, linestyles='dashed', color="red")
plt.plot(times, strain)
usual_plot(title=name_of_event, save_path=f"{save_results_path}{name_of_event}_event_strain")


# times, strain, center_in_seconds = get_strain_data(gw_name='GW150914', duration=32, unpack=True, version=3, sample_rate=4096,
#                                 center_at=1126259462.4)

# plt.plot(times, strain)
# usual_plot()


k, ps, windows = calculate_welch_average(x=times, y=strain, L=2)
mask = (k > 0)
plt.loglog(k[mask], ps[mask])
usual_plot()

for i in range(len(windows)):

    current_window = windows[i]
    current_time = current_window[0]
    current_strain = current_window[1]

    # print("min x max ", current_time.min(), marker, current_time.max())
    if current_time.min() < marker < current_time.max():

        my_whitened = whiten(y=current_strain, amp=jnp.sqrt(ps))
        my_bandpassed = bandpass(x=current_time, y=my_whitened)

        plt.plot(current_time, my_bandpassed)
        usual_plot(title=f"window {i}")


        xi_d_tilde = solve_data_equation_for_xi(data=current_strain, ps=ps)

        plt.plot(k, xi_d_tilde)
        tl = "Time window: " + str(np.float64(current_time.min())) + " - " + str(np.float64(current_time.max())) + f" (No. {i} 0-based)"
        usual_plot(title=tl)

        wigner_function, t, f = Stress_jft(xi=xi_d_tilde, time=current_time)
        smoothed_wigner = smooth_matrix(wigner_function, smoothing_lvl=5, mode="gaussian")

        visualize_stress(wigner_function, rows=f, cols=t, smooth=False)
        visualize_stress(smoothed_wigner, rows=f, cols=t, smooth=False,  save_path=f"{save_results_path}{name_of_event}_event_smoothed_wigner.png",
                         xlim=(14.4,14.6), ylim=(-400, 400))



