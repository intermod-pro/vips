import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit
from scipy.linalg import svd
from scipy.optimize import least_squares

from utils import get_load_path, untwist_downconversion

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npzfile:
    num_averages = np.asscalar(npzfile['num_averages'])
    control_freq = np.asscalar(npzfile['control_freq'])
    readout_freq = np.asscalar(npzfile['readout_freq'])
    fast_readout_length = np.asscalar(npzfile['fast_readout_length'])
    normal_readout_length = np.asscalar(npzfile['normal_readout_length'])
    pi_pulse_length = np.asscalar(npzfile['pi_pulse_length'])
    pi2_pulse_length = np.asscalar(npzfile['pi2_pulse_length'])
    fast_readout_amp = np.asscalar(npzfile['fast_readout_amp'])
    normal_readout_amp = np.asscalar(npzfile['normal_readout_amp'])
    pi_pulse_amp = np.asscalar(npzfile['pi_pulse_amp'])
    pi2_pulse_amp = np.asscalar(npzfile['pi2_pulse_amp'])
    which_pulse = np.asscalar(npzfile['which_pulse'])
    template_fast = npzfile['template_fast']
    template_pi = npzfile['template_pi']
    template_pi2 = npzfile['template_pi2']
    sample_length = np.asscalar(npzfile['sample_length'])
    wait_decay = np.asscalar(npzfile['wait_decay'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    brick_freq = np.asscalar(npzfile['brick_freq'])
    brick_pwr = np.asscalar(npzfile['brick_pwr'])
    gen_freq = np.asscalar(npzfile['gen_freq'])
    gen_pwr = np.asscalar(npzfile['gen_pwr'])
    detuning = np.asscalar(npzfile['detuning'])
    nr_delays = np.asscalar(npzfile['nr_delays'])
    dt_delays = np.asscalar(npzfile['dt_delays'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

delay_array = np.arange(nr_delays) * dt_delays
I_port = np.zeros((2, nr_delays, 2), dtype=np.complex128)
Q_port = np.zeros((2, nr_delays, 2), dtype=np.complex128)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
idx_fft = int(round(readout_freq * (idx_span / 4e9)))

for kk in range(2):
    for jj in range(nr_delays):
        for ii in range(2):
            data_i = result[kk, jj, ii, 0, idx_start:idx_stop]
            data_q = result[kk, jj, ii, 1, idx_start:idx_stop]
            fft_i = np.fft.rfft(data_i) / idx_span
            fft_q = np.fft.rfft(data_q) / idx_span
            I_port[kk, jj, ii] = fft_i[idx_fft]
            Q_port[kk, jj, ii] = fft_q[idx_fft]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)
signal = H_sideband

xquad = signal.real
yquad = signal.imag
amp = np.abs(signal)
_phase = np.angle(signal)
# phase = np.unwrap(_phase)
phase = _phase

diff_signal = amp[:, :, 0] - amp[:, :, 1]


def model(t, offset, amplitude, phase, n0, kappa, chi, delta, gamma):
    tau = (1 - np.exp(-(kappa + 2 * chi * 1j) * t)) / (kappa + 2 * chi * 1j)
    return offset + amplitude * np.imag(
        np.exp(-(gamma + delta * 1j) * t + (phase - 2 * n0 * chi * tau) * 1j))


def erf(params, t, measured, kappa, chi, delta, gamma):
    offset, amplitude, phase, n0 = params
    prediction = model(t, offset, amplitude, phase, n0, kappa, chi, delta, gamma)
    error = prediction - measured
    return error


def fit_all(x, y, kappa, chi, delta, gamma):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    first = (y[0] - offset) / amplitude
    if first > 1.:
        first = 1.
    elif first < -1.:
        first = -1.
    phase = np.arcsin(first)
    p0 = (
        offset,
        amplitude,
        phase,
        0.,
    )
    # popt, cov = curve_fit(
    #     model,
    #     x,
    #     y,
    #     p0=p0,
    #     # bounds=(
    #     #     [-np.inf, 0., 1e-9, 0., -np.pi],
    #     #     [np.inf, np.inf, 1e-3, np.inf, np.pi],
    #     # ),
    # )
    # perr = np.sqrt(np.diag(cov))
    # # offset, amplitude, T2, frequency, phase = popt
    res = least_squares(
        erf,
        x0=p0,
        args=(x, y, kappa, chi, delta, gamma),
    )
    popt = res.x

    # calculate errors: from curve_fit sourcecode
    # https://github.com/scipy/scipy/blob/v1.4.1/scipy/optimize/minpack.py#L784
    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    ysize = len(res.fun)
    cost = 2 * res.cost  # res.cost is half sum of squares!
    s_sq = cost / (ysize - len(p0))
    pcov = pcov * s_sq
    perr = np.sqrt(np.diag(pcov))

    return popt, perr


kappa = 2. * np.pi * 904e3
chi = -0.286 * kappa
delta = 2. * np.pi * 10e6
gamma = 1 / 10e-6
popt_g, perr_g = fit_all(delay_array, diff_signal[0, :], kappa, chi, delta, gamma)
popt_e, perr_e = fit_all(delay_array, diff_signal[1, :], kappa, chi, delta, gamma)

fig2, ax2 = plt.subplots(tight_layout=True)
ax2.plot(1e6 * delay_array, 1e6 * diff_signal[0, :], label="|g>")
ax2.plot(1e6 * delay_array, 1e6 * model(delay_array, *popt_g, kappa, chi, delta, gamma), ls='--', label=r"n0 = {:.2f}".format(popt_g[-1]))
ax2.plot(1e6 * delay_array, 1e6 * diff_signal[1, :], label="|e>")
ax2.plot(1e6 * delay_array, 1e6 * model(delay_array, *popt_e, kappa, chi, delta, gamma), ls='--', label=r"n0 = {:.2f}".format(popt_e[-1]))
ax2.set_xlabel(r"Ramsey delay [$\mathrm{\mu s}$]")
ax2.set_ylabel(r"Diff. amplitude [$\mathrm{\mu FS}$]")
ax2.set_title(which_pulse)
ax2.legend()
fig2.show()

print("Leftover |g>: {}".format(popt_g[-1]))
print("Leftover |e>: {}".format(popt_e[-1]))
