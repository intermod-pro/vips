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
    control_length = np.asscalar(npzfile['control_length'])
    wait_decay = np.asscalar(npzfile['wait_decay'])
    nr_delays = np.asscalar(npzfile['nr_delays'])
    dt_delays = np.asscalar(npzfile['dt_delays'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    brick_freq = np.asscalar(npzfile['brick_freq'])
    brick_pwr = np.asscalar(npzfile['brick_pwr'])
    gen_freq = np.asscalar(npzfile['gen_freq'])
    gen_pwr = np.asscalar(npzfile['gen_pwr'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

I_port = np.zeros(nr_delays, dtype=np.complex128)
Q_port = np.zeros(nr_delays, dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ii in range(nr_delays):
    data_i = result[ii, 0, idx_start:idx_stop]
    data_q = result[ii, 1, idx_start:idx_stop]
    fft_i = np.fft.rfft(data_i) / idx_span
    fft_q = np.fft.rfft(data_q) / idx_span
    I_port[ii] = fft_i[kk]
    Q_port[ii] = fft_q[kk]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)


def decay(t, *p):
    T1, xe, xg = p
    return xg + (xe - xg) * np.exp(-t / T1)


def fit_simple(t, x):
    T1 = 0.5 * (t[-1] - t[0])
    xe, xg = x[0], x[-1]
    p0 = (T1, xe, xg)
    popt, pcov = curve_fit(decay, t, x, p0)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


delay_array = np.arange(nr_delays) * dt_delays
popt, perr = fit_simple(delay_array, np.abs(H_sideband))
T1 = popt[0]
T1_err = perr[0]
print("T1 time: {} +- {} us".format(1e6 * T1, 1e6 * T1_err))


fig1, ax1 = plt.subplots(tight_layout=True)
ax1.plot(1e6 * delay_array, 1e6 * np.abs(H_sideband))
ax1.plot(1e6 * delay_array, 1e6 * decay(delay_array, *popt), '--')
ax1.set_ylabel(r"$A$ [$\mathrm{\mu FS}$]")
ax1.set_xlabel(r"Readout delay [$\mathrm{\mu s}$]")
fig1.show()


fig2, ax2 = plt.subplots(4, 1, sharex=True, tight_layout=True)
ax21, ax22, ax23, ax24 = ax2

ax21.plot(1e6 * delay_array, 1e6 * H_sideband.real)
ax22.plot(1e6 * delay_array, 1e6 * H_sideband.imag)
ax21.set_ylabel(r"$X$")
ax22.set_ylabel(r"$Y$")

ax23.plot(1e6 * delay_array, 1e6 * np.abs(H_sideband))
ax23.plot(1e6 * delay_array, 1e6 * decay(delay_array, *popt), '--')
ax24.plot(1e6 * delay_array, np.unwrap(np.angle(H_sideband)))
ax23.set_ylabel(r"$A$")
ax24.set_ylabel(r"$\phi$")

ax24.set_xlabel(r"Readout delay [$\mathrm{\mu s}$]")
fig2.show()
