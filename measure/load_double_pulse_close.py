import matplotlib.pyplot as plt
import numpy as np

from utils import get_load_path, untwist_downconversion
from utils import demodulate, demodulate_time

load_path = None
if load_path is None:
    load_path = get_load_path(__file__)

with np.load(load_path) as npzfile:
    num_averages = np.asscalar(npzfile['num_averages'])
    control_freq = np.asscalar(npzfile['control_freq'])
    readout_freq = np.asscalar(npzfile['readout_freq'])
    readout_length = np.asscalar(npzfile['readout_length'])
    control_length = np.asscalar(npzfile['control_length'])
    readout_amp = np.asscalar(npzfile['readout_amp'])
    control_amp = np.asscalar(npzfile['control_amp'])
    template_arr = npzfile['template_arr']
    template_pi = npzfile['template_pi']
    sample_length = np.asscalar(npzfile['sample_length'])
    wait_decay = np.asscalar(npzfile['wait_decay'])
    readout_sample_delay = np.asscalar(npzfile['readout_sample_delay'])
    brick_freq = np.asscalar(npzfile['brick_freq'])
    brick_pwr = np.asscalar(npzfile['brick_pwr'])
    gen_freq = np.asscalar(npzfile['gen_freq'])
    gen_pwr = np.asscalar(npzfile['gen_pwr'])
    result = npzfile['result']
    sourcecode = npzfile['sourcecode']

fsample = 4e+9
nr_samples = int(round(sample_length * fsample))
t_array = np.arange(nr_samples) / fsample
idx_fft = int(round(readout_freq * sample_length))
# dem_bw = idx_fft
dem_bw = int(round(190e6 * sample_length))

nr_templates = len(template_arr)
envelope_I = np.zeros((2, nr_templates, dem_bw), dtype=np.complex128)
envelope_Q = np.zeros((2, nr_templates, dem_bw), dtype=np.complex128)
for jj in range(2):
    for ii in range(nr_templates):
        envelope_I[jj, ii, :] = demodulate(result[jj, ii, 0, :], idx_fft, dem_bw)
        envelope_Q[jj, ii, :] = demodulate(result[jj, ii, 1, :], idx_fft, dem_bw)
envelope_L, envelope_H = untwist_downconversion(envelope_I, envelope_Q)

envelope = envelope_H
t_envelope = demodulate_time(t_array, dem_bw)

fig1, ax1 = plt.subplots(nr_templates, 1, sharex=True, sharey=True, tight_layout=True)

for ii in range(nr_templates):
    ax1[ii].plot(1e9 * t_array, 1e3 * result[0, ii, 0])
    ax1[ii].plot(1e9 * t_array, 1e3 * result[1, ii, 0])

ax1[-1].set_xlabel(r"Time [$\mathrm{ns}$]")
for _ax in ax1:
    _ax.set_ylabel(r"Voltage [$\mathrm{m FS}$]")
fig1.show()


fig2, ax2 = plt.subplots(nr_templates, 1, sharex=True, sharey=True, tight_layout=True)
for ii in range(nr_templates):
    ax2[ii].axhline(0, ls='--', c='tab:gray')
    ax2[ii].plot(1e9 * t_envelope, 1e3 * np.abs(envelope[0, ii, :]), label="|g>")
    ax2[ii].plot(1e9 * t_envelope, 1e3 * np.abs(envelope[1, ii, :]), label="|e>")
ax2[0].legend()
ax2[-1].set_xlabel(r"Time [$\mathrm{ns}$]")
for _ax in ax2:
    _ax.set_ylabel(r"$A$ [$\mathrm{m FS}$]")
fig2.show()


fig3, ax3 = plt.subplots(nr_templates, 1, sharex=True, sharey=True, tight_layout=True)
for ii in range(nr_templates):
    ax3[ii].axhline(0, ls='--', c='tab:gray')
    ax3[ii].plot(1e9 * t_envelope, np.angle(envelope[0, ii, :]), label="|g>")
    ax3[ii].plot(1e9 * t_envelope, np.angle(envelope[1, ii, :]), label="|e>")
ax3[0].legend()
ax3[-1].set_xlabel(r"Time [$\mathrm{ns}$]")
for _ax in ax3:
    _ax.set_ylim(-np.pi, np.pi)
    _ax.set_ylabel(r"$\phi$ [$\mathrm{rad}$]")
fig3.show()


fig4, ax4 = plt.subplots(1, 1, tight_layout=True)
ax4.axhline(0, ls='--', c='tab:gray')
for ii in range(nr_templates):
    ax4.plot(1e9 * t_envelope, 1e3 * np.abs(envelope[0, ii, :]), label="")
# ax4.legend()
ax4.set_xlabel(r"Time [$\mathrm{ns}$]")
ax4.set_ylabel(r"$A$ [$\mathrm{m FS}$]")
fig4.show()
