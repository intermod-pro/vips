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
    control_amp = np.asscalar(npzfile['control_amp'])
    sample_length = np.asscalar(npzfile['sample_length'])
    rabi_n = np.asscalar(npzfile['rabi_n'])
    rabi_dt = np.asscalar(npzfile['rabi_dt'])
    rabi_readout_delay = np.asscalar(npzfile['rabi_readout_delay'])
    rabi_decay = np.asscalar(npzfile['rabi_decay'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

I_port = np.zeros(rabi_n, dtype=np.complex128)
Q_port = np.zeros(rabi_n, dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ii in range(rabi_n):
    data_i = result[ii, 0, idx_start:idx_stop]
    data_q = result[ii, 1, idx_start:idx_stop]
    fft_i = np.fft.rfft(data_i) / idx_span
    fft_q = np.fft.rfft(data_q) / idx_span
    I_port[ii] = fft_i[kk]
    Q_port[ii] = fft_q[kk]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)

delay_array = rabi_dt * np.arange(rabi_n)


def func(t, offset, amplitude, T2, period, phase):
    frequency = 1 / period
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
    period = 1 / frequency
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
        period,
        phase,
    )
    popt, pcov = curve_fit(func, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    offset, amplitude, T2, period, phase = popt
    return popt, perr


# fit_x = fit_period(delay_array, H_sideband.real)
# fit_y = fit_period(delay_array, H_sideband.imag)
popt, perr = fit_period(delay_array, np.abs(H_sideband))
# fit_p = fit_period(delay_array, np.unwrap(np.angle(H_sideband)))

period = popt[3]
period_err = perr[3]
pi_len = round(period / 2 / 2e-9) * 2e-9
pi_2_len = round(period / 4 / 2e-9) * 2e-9
print("Rabi period: {} +- {} ns".format(1e9 * period, 1e9 * period_err))
print("Pi pulse length: {:.0f} ns".format(1e9 * pi_len))
print("Pi/2 pulse length: {:.0f} ns".format(1e9 * pi_2_len))


fig1, ax1 = plt.subplots(tight_layout=True)
ax1.plot(1e6 * delay_array, 1e6 * np.abs(H_sideband))
ax1.plot(1e6 * delay_array, 1e6 * func(delay_array, *popt), '--')
ax1.set_ylabel(r"$A$ [$\mathrm{\mu FS}$]")
ax1.set_xlabel(r"Pulse length [$\mathrm{\mu s}$]")
fig1.show()


fig2, ax2 = plt.subplots(4, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23, ax24 = ax2

# ax21.plot(1e6 * delay_array, amps_i, label='I')
# ax21.plot(1e6 * delay_array, func(delay_array, *fit_ai), '--', label='fit')
# ax21.plot(1e6 * delay_array, amps_q, label='Q')
# ax21.plot(1e6 * delay_array, func(delay_array, *fit_aq), '--', label='fit')
# ax22.plot(1e6 * delay_array, phases_i, label='I')
# ax22.plot(1e6 * delay_array, func(delay_array, *fit_pi), '--', label='fit')
# ax22.plot(1e6 * delay_array, phases_q, label='Q')
# ax22.plot(1e6 * delay_array, func(delay_array, *fit_pq), '--', label='fit')
# ax21.set_ylabel("Amplitude [arb. units]")
# ax22.set_ylabel("Phase [rad]")

ax21.plot(1e6 * delay_array, 1e6 * H_sideband.real)
ax22.plot(1e6 * delay_array, 1e6 * H_sideband.imag)
ax21.set_ylabel(r"$X$")
ax22.set_ylabel(r"$Y$")

ax23.plot(1e6 * delay_array, 1e6 * np.abs(H_sideband))
ax23.plot(1e6 * delay_array, 1e6 * func(delay_array, *popt), '--')
ax24.plot(1e6 * delay_array, np.unwrap(np.angle(H_sideband)))
ax23.set_ylabel(r"$A$")
ax24.set_ylabel(r"$\phi$")

ax24.set_xlabel(r"Pulse length [$\mathrm{\mu s}$]")
fig2.show()
