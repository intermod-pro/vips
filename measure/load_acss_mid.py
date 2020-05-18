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
    control_length = np.asscalar(npzfile['control_length'])
    control_shape = np.asscalar(npzfile['control_shape'])
    readout_amp = np.asscalar(npzfile['readout_amp'])
    control_amp = np.asscalar(npzfile['control_amp'])
    sample_length = np.asscalar(npzfile['sample_length'])
    nr_delays = np.asscalar(npzfile['nr_delays'])
    dt_delays = np.asscalar(npzfile['dt_delays'])
    wait_decay = np.asscalar(npzfile['wait_decay'])
    cavity_ringup = np.asscalar(npzfile['cavity_ringup'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    amp_array = npzfile['amp_array']
    qubit_freq = np.asscalar(npzfile['qubit_freq'])
    detuning = np.asscalar(npzfile['detuning'])
    brick_freq = np.asscalar(npzfile['brick_freq'])
    brick_pwr = np.asscalar(npzfile['brick_pwr'])
    gen_freq = np.asscalar(npzfile['gen_freq'])
    gen_pwr = np.asscalar(npzfile['gen_pwr'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

delay_array = dt_delays * np.arange(nr_delays)
nr_amps = len(amp_array)

I_port = np.zeros((nr_amps, nr_delays, 2), dtype=np.complex128)
Q_port = np.zeros((nr_amps, nr_delays, 2), dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
idx_fft = int(round(readout_freq * (idx_span / 4e9)))

for kk in range(nr_amps):
    for jj in range(2):  # +pi/2 and -pi/2
        for ii in range(nr_delays):
            data_i = result[kk, ii, jj, 0, idx_start:idx_stop] / num_averages
            data_q = result[kk, ii, jj, 1, idx_start:idx_stop] / num_averages
            fft_i = np.fft.rfft(data_i) / idx_span
            fft_q = np.fft.rfft(data_q) / idx_span
            I_port[kk, ii, jj] = fft_i[idx_fft]
            Q_port[kk, ii, jj] = fft_q[idx_fft]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)

# signal = H_sideband
# diff_signal = np.zeros((n_amps, nr_delays), dtype=np.complex128)
signal = np.abs(H_sideband)
diff_signal = np.zeros((nr_amps, nr_delays), dtype=np.float64)
for kk in range(nr_amps):
    diff_signal[kk, :] = signal[kk, :, 0] - signal[kk, :, 1]

# xquad = signal.real
# yquad = signal.imag
# amp = np.abs(signal)
# _phase = np.angle(signal)
# phase = np.unwrap(np.unwrap(_phase, axis=0), axis=1)

fig1, ax1 = plt.subplots(tight_layout=True)

ax1.plot(1e6 * delay_array, signal[0, :, 0])
ax1.plot(1e6 * delay_array, signal[0, :, 1])
ax1.set_ylabel(r'Amplitude $A$')
ax1.set_xlabel(r'Ramsey delay $\Delta t$ [$\mathrm{\mu s}$]')
fig1.show()

fig2, ax2 = plt.subplots(tight_layout=True)
for kk, amp in enumerate(amp_array):
    ax2.plot(1e6 * delay_array, diff_signal[kk, :], label=str(amp))
ax2.legend()
ax2.set_ylabel(r'Amplitude $A_\mathrm{diff}$')
ax2.set_xlabel(r'Ramsey delay $\Delta t$ [$\mathrm{\mu s}$]')
fig2.show()


def func(t, offset, amplitude, gamma, omega, phase):
    return offset + amplitude * np.exp(-gamma * t) * np.cos(omega * t + phase)


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
        1. / T2,
        2. * np.pi * frequency,
        phase,
    )
    popt, cov = curve_fit(func, x, y, p0=p0)
    perr = np.sqrt(np.diag(cov))
    offset, amplitude, gamma, omega, phase = popt
    return popt, perr


power = amp_array**2
omegas = np.zeros(nr_amps)
gammas = np.zeros(nr_amps)

for kk in range(nr_amps):
    popt, perr = fit_period(delay_array, diff_signal[kk, :])
    omegas[kk] = popt[3]
    gammas[kk] = popt[2]

pfit_omegas = np.polyfit(power, omegas, 1)
pfit_gammas = np.polyfit(power, gammas, 1)

fig3, ax3 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax31, ax32 = ax3
ax31.plot(1e6 * power, omegas / (2 * np.pi) / 1e3, '.')
ax31.plot(1e6 * power, np.polyval(pfit_omegas, power) / (2 * np.pi) / 1e3, '--')
ax32.plot(1e6 * power, gammas / 1e3, '.')
ax32.plot(1e6 * power, np.polyval(pfit_gammas, power) / 1e3, '--')
ax32.set_xlabel(r"Drive power [$\mathrm{\mu FS}$]")
ax31.set_ylabel(r"AC-Stark shift $\omega_\mathrm{S}/2\pi$ [$\mathrm{kHz}$]")
ax32.set_ylabel(r"Dephasing $\Gamma_\mathrm{M}$ [$\mathrm{kHz}$]")
fig3.show()


chi_kappa = pfit_gammas[0] / pfit_omegas[0] / 4

kappa = 2. * np.pi * 904e3  # from circle fit
chi = chi_kappa * kappa

ph_n = (np.polyval(pfit_omegas, power) - pfit_omegas[1]) / 2 / chi

fig4, ax4 = plt.subplots(tight_layout=True)
ax4.plot(1e6 * power, ph_n, '.-')
ax4.set_xlabel(r"Drive power [$\mathrm{\mu FS}$]")
ax4.set_ylabel(r"Avg. photon number $\left< n \right>$")
fig4.show()
