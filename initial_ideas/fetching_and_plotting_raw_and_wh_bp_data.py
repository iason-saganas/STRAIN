from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

h1 = TimeSeries.fetch_open_data('H1', 1126259457, 1126259467)
l1 = TimeSeries.fetch_open_data('L1', 1126259457, 1126259467)

h1b = h1.bandpass(50, 250).notch(60).notch(120)
l1b = l1.bandpass(50, 250).notch(60).notch(120)
l1b.shift('6.9ms')
l1b *= -1

plot = Plot(figsize=(12, 4))
ax = plot.add_subplot(xscale='auto-gps')
ax.plot(h1b, color='gwpy:ligo-hanford', label='LIGO-Hanford')
ax.plot(l1b, color='gwpy:ligo-livingston', label='LIGO-Livingston')
ax.set_epoch(1126259462.427)
ax.set_xlim(1126259462.2, 1126259462.5)
ax.set_ylim(-1e-21, 1e-21)
ax.set_ylabel('Strain wavelet?')
ax.legend()
plot.show()

plot = Plot(figsize=(12, 4))
ax = plot.add_subplot(xscale='auto-gps')
ax.plot(h1, color='gwpy:ligo-hanford', label='LIGO-Hanford')
ax.plot(l1, color='gwpy:ligo-livingston', label='LIGO-Livingston')
ax.set_epoch(1126259462.427)
ax.set_xlim(1126259462.2, 1126259462.5)
# ax.set_ylim(-1e-21, 1e-21)
ax.set_ylabel('Strain data')
ax.legend()
plot.show()

