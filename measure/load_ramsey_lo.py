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
    control_shape = np.asscalar(npzfile['control_shape'])
    wait_decay = np.asscalar(npzfile['wait_decay'])
    nr_delays = np.asscalar(npzfile['nr_delays'])
    dt_delays = np.asscalar(npzfile['dt_delays'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    brick_freq = np.asscalar(npzfile['brick_freq'])
    brick_pwr = np.asscalar(npzfile['brick_pwr'])
    gen_freq = np.asscalar(npzfile['gen_freq'])
    gen_pwr = np.asscalar(npzfile['gen_pwr'])
    detuning_array = npzfile['detuning_array']
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

nr_freqs = len(detuning_array)
delay_array = np.arange(nr_delays) * dt_delays
result.shape = (nr_freqs, nr_delays, 2, 2, result.shape[-1])
I_port = np.zeros((2, nr_freqs, nr_delays), dtype=np.complex128)
Q_port = np.zeros((2, nr_freqs, nr_delays), dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ll in range(2):
    for jj in range(nr_freqs):
        for ii in range(nr_delays):
            data_i = result[jj, ii, ll, 0, idx_start:idx_stop]
            data_q = result[jj, ii, ll, 1, idx_start:idx_stop]
            fft_i = np.fft.rfft(data_i) / idx_span
            fft_q = np.fft.rfft(data_q) / idx_span
            I_port[ll, jj, ii] = fft_i[kk]
            Q_port[ll, jj, ii] = fft_q[kk]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)

xquad = H_sideband.real
yquad = H_sideband.imag
amp = np.abs(H_sideband)
phase = np.angle(H_sideband)
# phase = np.unwrap(np.unwrap(_phase, axis=0), axis=1)


def myplot(quantity, label):
    cutoff = 1
    lowlim = np.percentile(quantity, cutoff / 2)
    higlim = np.percentile(quantity, 100 - cutoff / 2)
    fig, ax = plt.subplots(tight_layout=True)
    im = ax.imshow(
        quantity,
        aspect='auto',
        interpolation='none',
        origin='lower',
        extent=(
            1e6 * delay_array[0],
            1e6 * delay_array[-1],
            1e-3 * detuning_array[0],
            1e-3 * detuning_array[-1],
        ),
        vmin=lowlim,
        vmax=higlim,
    )
    cb = fig.colorbar(im)
    ax.set_xlabel(r'Ramsey delay $\Delta t$ [$\mathrm{\mu s}$]')
    ax.set_ylabel(r'Qubit detuning $\Delta f$ [$\mathrm{kHz}$]')
    cb.set_label(label)
    fig.show()
    return fig


signal_plus = amp[0]
signal_minus = amp[1]
signal_diff = amp[0] - amp[1]
fig1 = myplot(1e6 * signal_plus,
              label=r'Amplitude $A^\mathrm{+}$ [$\mathrm{\mu FS}$]')
fig2 = myplot(1e6 * signal_minus,
              label=r'Amplitude $A^\mathrm{-}$ [$\mathrm{\mu FS}$]')
fig3 = myplot(1e6 * signal_diff,
              label=r'Amplitude $A_\mathrm{diff}$ [$\mathrm{\mu FS}$]')


def func(t, offset, amplitude, T2, frequency, phase):
    return offset + amplitude * np.exp(
        -t / T2) * np.cos(2. * np.pi * frequency * t + phase)


def fit_simple(x, y):
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
    popt, cov = curve_fit(
        func,
        x,
        y,
        p0=p0,
        # bounds=(
        #     [-np.inf, 0., 1e-9, 0., -np.pi],
        #     [np.inf, np.inf, 1e-3, np.inf, np.pi],
        # ),
    )
    offset, amplitude, T2, frequency, phase = popt
    return popt


fit_T2 = np.zeros(nr_freqs)
fit_freq = np.zeros(nr_freqs)
for jj in range(nr_freqs):
    try:
        res = fit_simple(delay_array, signal_diff[jj])
        fit_T2[jj] = res[2]
        fit_freq[jj] = np.abs(res[3])
    except Exception:
        fit_T2[jj] = np.nan
        fit_freq[jj] = np.nan

fit_freq[fit_T2 > 1e-3] = np.nan
fit_T2[fit_T2 > 1e-3] = np.nan
fit_freq[fit_T2 < 1e-7] = np.nan
fit_T2[fit_T2 < 1e-7] = np.nan

n_fit = nr_freqs // 4
pfit1 = np.polyfit(detuning_array[:n_fit], fit_freq[:n_fit], 1)
pfit2 = np.polyfit(detuning_array[-n_fit:], fit_freq[-n_fit:], 1)
x0 = np.roots(pfit1 - pfit2)[0]

fig5, ax5 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax51, ax52 = ax5
ax51.semilogy(1e-3 * detuning_array, fit_T2, '.')
ax52.plot(1e-3 * detuning_array, 1e-3 * fit_freq, '.')
ax51.set_ylabel(r"Fitted $T_2$ [$\mathrm{s}$]")
ax52.set_ylabel(r"Fitted $\Delta f$ [$\mathrm{kHz}$]")
ax52.set_xlabel(r"Qubit detuning $\Delta f$ [$\mathrm{kHz}$]")
fig5.show()
_lims = ax52.axis()
ax52.plot(1e-3 * detuning_array,
          1e-3 * np.polyval(pfit1, detuning_array),
          '--',
          c='tab:gray')
ax52.plot(1e-3 * detuning_array,
          1e-3 * np.polyval(pfit2, detuning_array),
          '--',
          c='tab:gray')
ax52.axhline(0., ls='--', c='tab:gray')
ax52.axvline(1e-3 * x0, ls='--', c='tab:gray')
ax52.axis(_lims)
fig5.canvas.draw()
