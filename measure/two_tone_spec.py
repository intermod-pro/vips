import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..\\server")
import simple_lockin

sys.path.append('C:\\IMP Sessions and Settings\\settings\\startup_scripts')
from Periphery import Periphery

save_folder = "F:\\rfzynq_data"

center = 300e6
span = 10e6
npoints = 1024
bandwidth = 1e3
Nget = 1_100
Navg = 1_000

cavity_f0 = 6_029_496_000.
cavity_freq = 400e6
brick_freq = cavity_f0 - cavity_freq
brick_pwr = 7.5  # dBm
cavity_amp = 0.03125
cavity_ampI = cavity_amp
cavity_ampQ = cavity_amp  # * (1 + g12)
cavity_phaseI = 0.
cavity_phaseQ = +np.pi / 2  # + t12

# gen_freq = 4_049_435_192.
gen_freq = 4_049_390_300.
gen_pwr = 19.0
qubit_amp = 0.1
qubit_ampI = qubit_amp
qubit_ampQ = qubit_amp  # * (1 + g12)
qubit_phaseI = 0.
qubit_phaseQ = -np.pi / 2  # + t12


if npoints is not None:
    assert span / npoints > bandwidth

# Program local oscillator
p = Periphery()

# cavity drive: readout
Brick = Periphery.create_pinstrument(p, 'Brick', 'Vaunix_lab_brick', '0')
Brick.on()
Brick.set_power(brick_pwr)
Brick.set_external_reference()
Brick.set_frequency(brick_freq)
Source = Periphery.create_pinstrument(
    p, 'Source', 'Agilent_E8247C', '192.168.18.104')
Source.on()
Source.set_frequency(gen_freq / 1e9)
Source.set_power(gen_pwr)

with simple_lockin.Lockin() as lockin:
    if npoints is not None:
        _freq_array = np.linspace(center - span / 2, center + span / 2,
                                  npoints)
        qubit_freq_array, df = lockin.tune(_freq_array, bandwidth)
    else:
        _, df = lockin.tune(center, bandwidth)
        ncenter = int(round(center / df))
        nspan = int(round(span / df))
        narray = np.arange(-nspan // 2, nspan // 2) + ncenter
        qubit_freq_array = df * narray
        npoints = len(qubit_freq_array)
        print(npoints)

    resp_I = np.zeros(npoints, np.complex128)
    resp_Q = np.zeros(npoints, np.complex128)

    lockin.set_df(df)
    lockin.set_frequencies([cavity_freq, qubit_freq_array[0]])
    lockin.set_amplitudes([cavity_ampI, 0.], port=1)
    lockin.set_amplitudes([cavity_ampQ, 0.], port=2)
    lockin.set_amplitudes([0., qubit_ampI], port=3)
    lockin.set_amplitudes([0., qubit_ampQ], port=4)
    lockin.set_phases([cavity_phaseI, 0.], port=1)
    lockin.set_phases([cavity_phaseQ, 0.], port=2)
    lockin.set_phases([0., qubit_phaseI], port=3)
    lockin.set_phases([0., qubit_phaseQ], port=4)

    lockin.apply_settings()
    time.sleep(1)

    # data1 = lockin.get_time_data(1)
    # data2 = lockin.get_time_data(2)
    # assert False

    for ii, qubit_freq in enumerate(qubit_freq_array):
        print(ii)
        lockin.set_frequencies([cavity_freq, qubit_freq])
        lockin.apply_settings()

        lockin.start_lockin()
        pixels = lockin.get_pixels(Nget)
        lockin.stop_lockin()

        pix_avg = np.mean(pixels[-Navg:], axis=0)
        resp_I[ii] = pix_avg[0, 0]
        resp_Q[ii] = pix_avg[1, 0]

        # data = lockin.get_time_data(port=1, ns=lockin.get_ns() * Navg)
        # data_fft = np.fft.rfft(data) / len(data)

        # kk = int(round(freq / df)) * Navg
        # amp_array[ii] = np.abs(data_fft[kk])
        # phase_array[ii] = np.angle(data_fft[kk])

    lockin.set_amplitudes(0.)
    lockin.apply_settings()

Source._visainstrument.close()

fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax11, ax12 = ax1
ax11.semilogy(qubit_freq_array, np.abs(resp_I))
ax11.semilogy(qubit_freq_array, np.abs(resp_Q))
ax12.plot(qubit_freq_array, np.angle(resp_I))
ax12.plot(qubit_freq_array, np.angle(resp_Q))
fig1.show()

with open(__file__, mode='rt', encoding='utf-8') as f:
    sourcecode = f.readlines()
scriptname = os.path.splitext(os.path.basename(__file__))[0]
save_filename = "{:s}_{:s}.npz".format(
    scriptname,
    time.strftime("%Y%m%d_%H%M%S"),
)
save_path = os.path.join(save_folder, save_filename)

np.savez(
    save_path,
    brick_freq=brick_freq,
    gen_freq=gen_freq,
    brick_pwr=brick_pwr,
    gen_pwr=gen_pwr,
    cavity_amp=cavity_amp,
    qubit_amp=qubit_amp,
    cavity_freq=cavity_freq,
    qubit_freq_array=qubit_freq_array,
    resp_I=resp_I,
    resp_Q=resp_Q,
    df=df,
    Navg=Navg,
    sourcecode=sourcecode,
)
