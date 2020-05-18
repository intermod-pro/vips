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

center = 250e6
span = 300e6
nr_freqs = 512
bandwidth = 1e2
Nget = 22
Navg = 20

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
# gen_freq = 4_049_390_300.
gen_freq = 3_881_890_300.
gen_pwr = 19.0
qubit_amp_array = np.logspace(-10, 0, 256, base=2)
# qubit_amp_array = np.logspace(-10, 0, 4, base=2)
nr_amps = len(qubit_amp_array)
qubit_phaseI = 0.
qubit_phaseQ = -np.pi / 2  # + t12

if nr_freqs is not None:
    assert span / nr_freqs > bandwidth

# Program local oscillator
p = Periphery()

# cavity drive: readout
Brick = Periphery.create_pinstrument(p, 'Brick', 'Vaunix_lab_brick', '0')
Brick.on()
Brick.set_power(brick_pwr)
Brick.set_external_reference()
Brick.set_frequency(brick_freq)
Source = Periphery.create_pinstrument(p, 'Source', 'Agilent_E8247C',
                                      '192.168.18.104')
Source.on()
Source.set_frequency(gen_freq / 1e9)
Source.set_power(gen_pwr)


def format_sec(s):
    """ Utility function to format a time interval in seconds
    into a more human-readable string.

    Args:
        s (float): time interval in seconds

    Returns:
        (str): time interval in the form "X h Y m Z.z s"

    Examples:
        >>> format_sec(12345.6)
        '3h 25m 45.6s'
    """
    if s < 1.:
        return "{:.1f}ms".format(s * 1e3)

    h = int(s // 3600)
    s -= h * 3600.

    m = int(s // 60)
    s -= m * 60

    if h:
        res = "{:d}h {:d}m {:.1f}s".format(h, m, s)
    elif m:
        res = "{:d}m {:.1f}s".format(m, s)
    else:
        res = "{:.1f}s".format(s)

    return res


with simple_lockin.Lockin() as lockin:
    if nr_freqs is not None:
        _freq_array = np.linspace(center - span / 2, center + span / 2,
                                  nr_freqs)
        qubit_freq_array, df = lockin.tune(_freq_array, bandwidth)
    else:
        _, df = lockin.tune(center, bandwidth)
        ncenter = int(round(center / df))
        nspan = int(round(span / df))
        narray = np.arange(-nspan // 2, nspan // 2) + ncenter
        qubit_freq_array = df * narray
        nr_freqs = len(qubit_freq_array)
        print(nr_freqs)

    resp_I = np.zeros((nr_amps, nr_freqs), np.complex128)
    resp_Q = np.zeros((nr_amps, nr_freqs), np.complex128)

    t_start = time.time()
    for jj, qubit_amp in enumerate(qubit_amp_array):
        qubit_ampI = qubit_amp
        qubit_ampQ = qubit_amp  # * (1 + g12)

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
            resp_I[jj, ii] = pix_avg[0, 0]
            resp_Q[jj, ii] = pix_avg[1, 0]

            # data = lockin.get_time_data(port=1, ns=lockin.get_ns() * Navg)
            # data_fft = np.fft.rfft(data) / len(data)

            # kk = int(round(freq / df)) * Navg
            # amp_array[ii] = np.abs(data_fft[kk])
            # phase_array[ii] = np.angle(data_fft[kk])

            t_now = time.time()
            t_sofar = t_now - t_start
            nr_sofar = jj * nr_freqs + ii + 1
            nr_left = (nr_amps - jj - 1) * nr_freqs + (nr_freqs - ii - 1)
            t_avg = t_sofar / nr_sofar
            t_left = t_avg * nr_left
            str_left = format_sec(t_left)
            print("Time remaining: {:s}".format(str_left))

    lockin.set_amplitudes(0.)
    lockin.apply_settings()

Source._visainstrument.close()

resp_array = resp_I
resp_dB = 20. * np.log10(np.abs(resp_array))

cutoff = 1.  # %
lowlim = np.percentile(resp_dB, cutoff)
highlim = np.percentile(resp_dB, 100. - cutoff)

amp_dBFS = 20 * np.log10(qubit_amp_array / 1.0)
fig1, ax1 = plt.subplots(tight_layout=True)
im = ax1.imshow(
    resp_dB,
    origin='lower',
    aspect='auto',
    interpolation='none',
    # extent=(freq_array[0], freq_array[-1], amp_array[0], amp_array[-1]),
    extent=(qubit_freq_array[0], qubit_freq_array[-1], amp_dBFS[0],
            amp_dBFS[-1]),
    vmin=lowlim,
    vmax=highlim,
)
ax1.set_xlabel("Frequency [Hz]")
# ax1.set_ylabel("Drive amplitude [V]")
ax1.set_ylabel("Drive amplitude [dBFS]")
cb = fig1.colorbar(im)
cb.set_label("Response amplitude [dB]")
fig1.show()

# fig2, ax2 = plt.subplots(tight_layout=True)
# # ax2.plot(freq_array, resp_dB[0], label=str(amp_array[0]))
# for div in [64, 32, 16, 8, 4, 2]:
#     kk = Namps // div - 1
#     l, = ax2.plot(freq_array, resp_dB[kk], label=str(amp_array[kk]))
#     ax1.axhline(amp_array[kk], ls='--', c=l.get_color())
# ax2.set_xlabel("Frequency [Hz]")
# ax2.set_ylabel("Response amplitude [dB]")
# ax2.legend(title="Drive amplitude [V]")
# fig2.show()

fig1.canvas.draw()
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
    qubit_amp_array=qubit_amp_array,
    cavity_freq=cavity_freq,
    qubit_freq_array=qubit_freq_array,
    resp_I=resp_I,
    resp_Q=resp_Q,
    df=df,
    Navg=Navg,
    sourcecode=sourcecode,
)
