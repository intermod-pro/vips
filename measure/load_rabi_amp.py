import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import get_load_path, untwist_downconversion

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npzfile:
    num_averages = np.asscalar(npzfile['num_averages'])
    control_freq = np.asscalar(npzfile['control_freq'])
    readout_freq = np.asscalar(npzfile['readout_freq'])
    readout_length = np.asscalar(npzfile['readout_length'])
    readout_amp = np.asscalar(npzfile['readout_amp'])
    control_amp_array = npzfile['control_amp_array']
    sample_length = np.asscalar(npzfile['sample_length'])
    control_length = np.asscalar(npzfile['control_length'])
    rabi_readout_delay = np.asscalar(npzfile['rabi_readout_delay'])
    rabi_decay = np.asscalar(npzfile['rabi_decay'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

nr_amps = len(control_amp_array)
I_port = np.zeros(nr_amps, dtype=np.complex128)
Q_port = np.zeros(nr_amps, dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ii in range(nr_amps):
    data_i = result[ii, 0, idx_start:idx_stop]
    data_q = result[ii, 1, idx_start:idx_stop]
    fft_i = np.fft.rfft(data_i) / idx_span
    fft_q = np.fft.rfft(data_q) / idx_span
    I_port[ii] = fft_i[kk]
    Q_port[ii] = fft_q[kk]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)


def func(t, offset, amplitude, T2, frequency, phase):
    return offset + amplitude * np.exp(
        -t / T2) * np.cos(2. * np.pi * frequency * t + phase)


def fit_period(x, y):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    first = (y[0] - offset) / amplitude
    if first > 1.:
        first = 1.
    elif first < -1.:
        first = -1.
    phase = np.arccos(first)
    p0 = (
        offset,
        amplitude,
        T2,
        frequency,
        phase,
    )
    popt, cov = curve_fit(func, x, y, p0=p0)
    offset, amplitude, T2, frequency, phase = popt
    return popt


fit_a = fit_period(control_amp_array, np.abs(H_sideband))
period = 1. / fit_a[3]
print("Pi pulse amplitude: {}".format(period / 2))
print("Pi/2 pulse amplitude: {}".format(period / 4))


fig1, ax1 = plt.subplots(tight_layout=True)
ax1.plot(control_amp_array, 1e6 * np.abs(H_sideband))
ax1.plot(control_amp_array, 1e6 * func(control_amp_array, *fit_a), '--')
ax1.set_ylabel(r"$A$ [$\mathrm{\mu FS}$]")
ax1.set_xlabel(r"Pulse amplitude [FS]")
fig1.show()



fig2, ax2 = plt.subplots(4, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23, ax24 = ax2

ax21.plot(control_amp_array, 1e6 * H_sideband.real)
ax22.plot(control_amp_array, 1e6 * H_sideband.imag)
ax21.set_ylabel(r"$X$")
ax22.set_ylabel(r"$Y$")

ax23.plot(control_amp_array, 1e6 * np.abs(H_sideband))
ax23.plot(control_amp_array, 1e6 * func(control_amp_array, *fit_a), '--')
ax24.plot(control_amp_array, np.unwrap(np.angle(H_sideband)))
ax23.set_ylabel(r"$A$")
ax24.set_ylabel(r"$\phi$")

ax24.set_xlabel(r"Pulse amplitude [FS]")
fig2.show()
