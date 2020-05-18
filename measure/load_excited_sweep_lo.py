import matplotlib.pyplot as plt
import numpy as np
from resonator_tools import circuit

from utils import get_load_path, untwist_downconversion

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npzfile:
    num_averages = np.asscalar(npzfile['num_averages'])
    control_freq = np.asscalar(npzfile['control_freq'])
    readout_freq = np.asscalar(npzfile['readout_freq'])
    control_length = np.asscalar(npzfile['control_length'])
    readout_length = np.asscalar(npzfile['readout_length'])
    readout_amp = np.asscalar(npzfile['readout_amp'])
    control_amp = np.asscalar(npzfile['control_amp'])
    control_shape = np.asscalar(npzfile['control_shape'])
    sample_length = np.asscalar(npzfile['sample_length'])
    wait_decay = np.asscalar(npzfile['wait_decay'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    detuning_array = npzfile['detuning_array']
    brick_freq = np.asscalar(npzfile['brick_freq'])
    brick_pwr = np.asscalar(npzfile['brick_pwr'])
    gen_freq = np.asscalar(npzfile['gen_freq'])
    gen_pwr = np.asscalar(npzfile['gen_pwr'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

nr_freqs = len(detuning_array)
freq_arr = brick_freq + readout_freq + detuning_array

I_port = np.zeros((nr_freqs, 2), dtype=np.complex128)
Q_port = np.zeros((nr_freqs, 2), dtype=np.complex128)
# idx_start = 2200
# idx_stop = 3400
idx_start = 0
idx_stop = result.shape[-1]
idx_span = idx_stop - idx_start
idx_fft = int(round(readout_freq * (idx_span / 4e9)))

for kk in range(nr_freqs):
    for jj in range(2):
        data_i = result[kk, jj, 0, idx_start:idx_stop] / num_averages
        data_q = result[kk, jj, 1, idx_start:idx_stop] / num_averages
        fft_i = np.fft.rfft(data_i) / idx_span
        fft_q = np.fft.rfft(data_q) / idx_span
        I_port[kk, jj] = fft_i[idx_fft]
        Q_port[kk, jj] = fft_q[idx_fft]

L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)
signal = H_sideband

xquad_g = signal[:, 0].real
yquad_g = signal[:, 0].imag
amp_g = np.abs(signal[:, 0])
_phase_g = np.angle(signal[:, 0])
phase_g = np.unwrap(_phase_g)

xquad_e = signal[:, 1].real
yquad_e = signal[:, 1].imag
amp_e = np.abs(signal[:, 1])
_phase_e = np.angle(signal[:, 1])
phase_e = np.unwrap(_phase_e)

port_g = circuit.notch_port(freq_arr, signal[:, 0])
port_e = circuit.notch_port(freq_arr, signal[:, 1])
port_g.autofit()
port_e.autofit()
fr_g = port_g.fitresults['fr']
fr_e = port_e.fitresults['fr']
port_g.autofit(fcrop=(fr_g - 1e6, fr_g + 1e6))
port_e.autofit(fcrop=(fr_e - 1e6, fr_e + 1e6))
fr_g = port_g.fitresults['fr']
fr_e = port_e.fitresults['fr']

fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
ax1, ax2 = ax
ax1.semilogy(1e-9 * freq_arr, amp_g)
ax1.semilogy(1e-9 * freq_arr, amp_e)
ax1.semilogy(1e-9 * freq_arr, np.abs(port_g.z_data_sim), '--')
ax1.semilogy(1e-9 * freq_arr, np.abs(port_e.z_data_sim), '--')
ax2.plot(1e-9 * freq_arr, phase_g)
ax2.plot(1e-9 * freq_arr, phase_e)
ax2.plot(1e-9 * freq_arr, np.unwrap(np.angle(port_g.z_data_sim)), '--')
ax2.plot(1e-9 * freq_arr, np.unwrap(np.angle(port_e.z_data_sim)), '--')
fig.show()
