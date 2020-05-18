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

# center = 400e6
center = 250e6
# span = 10e6
span = 300e6
# npoints = 1024
npoints = 65536
bandwidth = 1e3
Nget = 110
Navg = 100

# brick_freq = 5.629e9  # Hz
# brick_freq = 5.917e9  # Hz
# brick_freq = 6.059e9  # Hz
# brick_freq = 6.209e9  # Hz
# brick_freq = 6.359e9  # Hz
brick_freq = 6.509e9  # Hz
brick_pwr = 7.5  # dBm

# amp_array = np.linspace(1. / 256, 1.0, 256)
amp = 1 / 32

# Mixer on ports 1-2
g12 = 0.080164
t12 = -0.127810
ampI = amp
ampQ = amp  # * (1 + g12)
phaseI = 0.
phaseQ = +np.pi / 2  # + t12

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

with simple_lockin.Lockin() as lockin:
    if npoints is not None:
        _freq_array = np.linspace(center - span / 2, center + span / 2,
                                  npoints)
        freq_array, df = lockin.tune(_freq_array, bandwidth)
    else:
        _, df = lockin.tune(center, bandwidth)
        ncenter = int(round(center / df))
        nspan = int(round(span / df))
        narray = np.arange(-nspan // 2, nspan // 2) + ncenter
        freq_array = df * narray
        npoints = len(freq_array)
        print(npoints)

    resp_I = np.zeros(npoints, np.complex128)
    resp_Q = np.zeros(npoints, np.complex128)

    lockin.set_df(df)
    lockin.set_frequencies(freq_array[0])
    lockin.set_amplitudes(ampI, port=1)
    lockin.set_amplitudes(ampQ, port=2)
    lockin.set_phases(phaseI, port=1)
    lockin.set_phases(phaseQ, port=2)

    lockin.apply_settings()
    time.sleep(1)

    # data1 = lockin.get_time_data(1)
    # data2 = lockin.get_time_data(2)
    # assert False

    for ii, freq in enumerate(freq_array):
        print(ii)
        lockin.set_frequencies(freq)
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

    lockin.set_amplitudes(0., port=1)
    lockin.set_amplitudes(0., port=2)
    lockin.apply_settings()

fig1, ax1 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax11, ax12 = ax1
ax11.semilogy(freq_array, np.abs(resp_I))
ax11.semilogy(freq_array, np.abs(resp_Q))
ax12.plot(freq_array, np.angle(resp_I))
ax12.plot(freq_array, np.angle(resp_Q))
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
    brick_pwr=brick_pwr,
    amp=amp,
    freq_array=freq_array,
    resp_I=resp_I,
    resp_Q=resp_Q,
    df=df,
    Navg=Navg,
    sourcecode=sourcecode,
)
