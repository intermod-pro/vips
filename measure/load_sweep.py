import os
import sys
import time

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
    amp = np.asscalar(npz['amp'])
    freq_array = npz['freq_array']
    resp_I = npz['resp_I']
    resp_Q = npz['resp_Q']
    df = np.asscalar(npz['df'])
    Navg = np.asscalar(npz['Navg'])
    sourcecode = npz['sourcecode']


L_sideband, H_sideband = untwist_downconversion(resp_I, resp_Q)

resp_array = H_sideband
resp_dB = 20. * np.log10(np.abs(resp_array))

actual_freq = brick_freq + freq_array


fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax11, ax12 = ax1
ax11.plot(1e-9 * actual_freq, resp_dB)
ax12.plot(1e-9 * actual_freq, np.angle(resp_array))
ax12.set_xlabel("Frequency [GHz]")
ax11.set_ylabel("Response amplitude [dB]")
ax12.set_ylabel("Response phase [rad]")
fig1.show()

# idx = np.argmin(np.abs(amp_dBFS - (-30.)))
# port = circuit.notch_port(actual_freq, resp_scaled[idx])
# port.autofit()
# _fr = port.fitresults['fr']
# port.autofit(fcrop=(_fr - 2e6, _fr + 2e6))
# fig3, ax3 = plt.subplots(2, 1, sharex=True, tight_layout=True)
# ax31, ax32 = ax3
# ax31.plot(actual_freq, resp_dB[idx])
# ax31.plot(port.f_data, 20 * np.log10(np.abs(port.z_data_sim)))
# ax32.plot(actual_freq, np.angle(resp_scaled[idx]))
# ax32.plot(port.f_data, np.angle(port.z_data_sim))
# fig3.show()
