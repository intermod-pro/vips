import matplotlib.pyplot as plt
import numpy as np

from resonator_tools import circuit

from utils import get_load_path, untwist_downconversion

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npz:
    brick_freq = np.asscalar(npz['brick_freq'])
    brick_pwr = np.asscalar(npz['brick_pwr'])
    amp_array = npz['amp_array']
    freq_array = npz['freq_array']
    resp_I = npz['resp_I']
    resp_Q = npz['resp_Q']
    df = np.asscalar(npz['df'])
    Navg = np.asscalar(npz['Navg'])
    sourcecode = npz['sourcecode']

nr_amps = len(amp_array)

L_sideband, H_sideband = untwist_downconversion(resp_I, resp_Q)

resp_array = H_sideband
resp_scaled = np.zeros_like(resp_array)
for jj in range(nr_amps):
    resp_scaled[jj] = resp_array[jj] / amp_array[jj]

resp_dB = 20. * np.log10(np.abs(resp_scaled))

cutoff = 1.  # %
lowlim = np.percentile(resp_dB, cutoff)
highlim = np.percentile(resp_dB, 100. - cutoff)

amp_dBFS = 20 * np.log10(amp_array / 1.0)
fig1, ax1 = plt.subplots(tight_layout=True)
im = ax1.imshow(
    resp_dB,
    origin='lower',
    aspect='auto',
    interpolation='none',
    # extent=(freq_array[0], freq_array[-1], amp_array[0], amp_array[-1]),
    extent=(freq_array[0], freq_array[-1], amp_dBFS[0], amp_dBFS[-1]),
    vmin=lowlim,
    vmax=highlim,
)
ax1.set_xlabel("Frequency [Hz]")
# ax1.set_ylabel("Drive amplitude [V]")
ax1.set_ylabel("Drive amplitude [dBFS]")
cb = fig1.colorbar(im)
cb.set_label("Response amplitude [dB]")
fig1.show()

fig2, ax2 = plt.subplots(tight_layout=True)
# ax2.plot(freq_array, resp_dB[0], label=str(amp_array[0]))
Nplots = 6
for ii in range(Nplots):
    kk = (ii + 1) * nr_amps // (Nplots + 1)
    l, = ax2.plot(freq_array, resp_dB[kk], label="{:.0f}".format(amp_dBFS[kk]))
    ax1.axhline(amp_dBFS[kk], ls='--', c=l.get_color())
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Response amplitude [dB]")
ax2.legend(title="Drive amplitude [dBFS]")
fig2.show()

fig1.canvas.draw()

actual_freq = brick_freq + freq_array
idx = np.argmin(np.abs(amp_dBFS - (-30.)))
port = circuit.notch_port(actual_freq, resp_scaled[idx])
port.autofit()
_fr = port.fitresults['fr']
# port.autofit(fcrop=(_fr - 1e6, _fr + 1e6))
fig3, ax3 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax31, ax32 = ax3
ax31.plot(actual_freq, resp_dB[idx])
ax31.plot(port.f_data, 20 * np.log10(np.abs(port.z_data_sim)))
ax32.plot(actual_freq, np.angle(resp_scaled[idx]))
ax32.plot(port.f_data, np.angle(port.z_data_sim))
fig3.show()
