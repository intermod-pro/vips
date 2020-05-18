import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

save_folder = "F:\\rfzynq_data"
load_filename = None

if load_filename is None:
    try:
        load_filename = sys.argv[1]
    except Exception:
        print("No or invalid filename specified...")

        all_files = sorted(os.listdir(save_folder))
        my_files = [
            x for x in all_files
            if x.startswith("two_tone_spec_20") and x.endswith(".npz")
        ]
        if my_files:
            load_filename = my_files[-1]
            print("Falling back to {:s}".format(load_filename))
        else:
            print("No valid file found")
            sys.exit()

load_path = os.path.join(save_folder, load_filename)
with np.load(load_path) as npz:
    brick_freq = np.asscalar(npz['brick_freq'])
    gen_freq = np.asscalar(npz['gen_freq'])
    brick_pwr = np.asscalar(npz['brick_pwr'])
    gen_pwr = np.asscalar(npz['gen_pwr'])
    cavity_amp = np.asscalar(npz['cavity_amp'])
    qubit_amp = np.asscalar(npz['qubit_amp'])
    cavity_freq = np.asscalar(npz['cavity_freq'])
    qubit_freq_array = npz['qubit_freq_array']
    resp_I = npz['resp_I']
    resp_Q = npz['resp_Q']
    df = np.asscalar(npz['df'])
    Navg = np.asscalar(npz['Navg'])
    sourcecode = npz['sourcecode']


def untwist_downconversion(I_port, Q_port):
    L_sideband = np.zeros_like(I_port)
    H_sideband = np.zeros_like(Q_port)

    L_sideband.real += I_port.real - Q_port.imag
    L_sideband.imag += Q_port.real + I_port.imag
    H_sideband.real += I_port.real + Q_port.imag
    H_sideband.imag += -(Q_port.real - I_port.imag)

    L_sideband.imag *= -1

    return L_sideband, H_sideband


L_sideband, H_sideband = untwist_downconversion(resp_I, resp_Q)

resp_array = H_sideband
resp_dB = 20. * np.log10(np.abs(resp_array))
actual_freq = gen_freq - qubit_freq_array

fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax11, ax12 = ax1
ax11.plot(actual_freq, resp_dB)
ax12.plot(actual_freq, np.angle(resp_array))
ax12.set_xlabel("Qubit-drive frequency [GHz]")
ax11.set_ylabel("Response amplitude [dBFS]")
ax12.set_ylabel("Response phase [rad]")
fig1.show()
