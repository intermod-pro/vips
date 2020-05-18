import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import get_load_path, untwist_downconversion

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npzfile:
    num_averages = np.asscalar(npzfile['num_averages'])
    control_freq_array = npzfile['control_freq_array']
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
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

nr_freqs = len(control_freq_array)
delay_array = np.arange(nr_delays) * dt_delays
result.shape = (nr_freqs, nr_delays, 2, result.shape[-1])
I_port = np.zeros((nr_freqs, nr_delays), dtype=np.complex128)
Q_port = np.zeros((nr_freqs, nr_delays), dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for jj in range(nr_freqs):
    for ii in range(nr_delays):
        data_i = result[jj, ii, 0, idx_start:idx_stop]
        data_q = result[jj, ii, 1, idx_start:idx_stop]
        fft_i = np.fft.rfft(data_i) / idx_span
        fft_q = np.fft.rfft(data_q) / idx_span
        I_port[jj, ii] = fft_i[kk]
        Q_port[jj, ii] = fft_q[kk]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)

xquad = H_sideband.real
yquad = H_sideband.imag
amp = np.abs(H_sideband)
_phase = np.angle(H_sideband)
phase = np.unwrap(np.unwrap(_phase, axis=0), axis=1)

detuning_array = control_freq_array - 300e6


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


# fig1 = myplot(xquad, label=r'$X$ quadrature [$\mathrm{ADU}$]')
# fig2 = myplot(yquad, label=r'$Y$ quadrature [$\mathrm{ADU}$]')
fig3 = myplot(amp, label=r'Amplitude $A$ [$\mathrm{ADU}$]')
# fig4 = myplot(phase, label=r'Phase $\phi$ [$\mathrm{ADU}$]')
"""
fit_T2 = np.zeros((4, n_freqs))
fit_freq = np.zeros((4, n_freqs))
for jj in range(n_freqs):
    for ii, quantity in enumerate([
            xquad[jj],
            # yquad[jj],
            amp[jj],
            # phase[jj],
    ]):
        try:
            res = fit_simple(delay_array[1:], quantity[1:])
            fit_T2[ii, jj] = res[2]
            fit_freq[ii, jj] = np.abs(res[3])
        except Exception:
            fit_T2[ii, jj] = np.nan
            fit_freq[ii, jj] = np.nan

fit_freq[fit_T2 > 1e-3] = np.nan
fit_T2[fit_T2 > 1e-3] = np.nan
fit_freq[fit_T2 < 1e-7] = np.nan
fit_T2[fit_T2 < 1e-7] = np.nan

fig5, ax5 = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax51, ax52 = ax5
labels = [
    r'$X$',
    # r'$Y$',
    r'$A$',
    # r'$\phi$',
]
for ii, label in enumerate(labels):
    ax51.semilogy(1e-3 * detuning_array, fit_T2[ii], '.', label=label)
    ax52.plot(1e-3 * detuning_array, 1e-3 * fit_freq[ii], '.', label=label)
ax51.legend(ncol=2)
ax52.legend(ncol=2)
ax51.set_ylabel(r"Fitted $T_2$ [$\mathrm{s}$]")
ax52.set_ylabel(r"Fitted $\Delta f$ [$\mathrm{kHz}$]")
ax52.set_xlabel(r"Qubit detuning $\Delta f$ [$\mathrm{kHz}$]")
fig5.show()
"""
