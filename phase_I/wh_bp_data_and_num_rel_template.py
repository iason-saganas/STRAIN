import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('MacOsX')

print(matplotlib.get_backend())

# data from https://gwosc.org/s/events/GW150914/GW150914.html

num_rel_data = np.loadtxt('../data/data_txt/fig1-waveform-H.txt')
time_num_rel, strain_num_rel = (num_rel_data[:, 0], num_rel_data[:, 1])

real_strain_data = np.loadtxt('../data/data_txt/fig1-observed-H.txt')
time, strain = (real_strain_data[:, 0], real_strain_data[:, 1])

plt.figure(figsize=(10,6))
plt.title("GW150419")
plt.xlabel("time (s)")
plt.ylabel("strain $[10^{-21}]$")
plt.plot(time_num_rel, strain_num_rel, "b-", markersize=0.5, label="Numerical relativity best fit template")
plt.plot(time, strain, "r-", markersize=0.5, label="Extensively cleaned data")
plt.legend()
plt.show()