import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import get_load_path, untwist_downconversion

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npzfile:
    num_averages = np.asscalar(npzfile['num_averages'])
    control_freq = npzfile['control_freq']
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

delay_array = np.arange(nr_delays) * dt_delays
# result.shape = (nr_delays, 2, result.shape[-1])
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
signal = H_sideband

xquad = signal.real
yquad = signal.imag
amp = np.abs(signal)
_phase = np.angle(signal)
phase = np.unwrap(_phase)


fig1, ax1 = plt.subplots(4, 1, sharex=True, tight_layout=True)
ax11, ax12, ax13, ax14 = ax1

ax11.plot(1e6 * delay_array, xquad)
ax12.plot(1e6 * delay_array, yquad)
ax11.set_ylabel(r"$X$")
ax12.set_ylabel(r"$Y$")

ax13.plot(1e6 * delay_array, amp)
ax14.plot(1e6 * delay_array, phase)
ax13.set_ylabel(r"$A$")
ax14.set_ylabel(r"$\phi$")

ax14.set_xlabel(r"Rabi delay [$\mathrm{\mu s}$]")
fig1.show()
