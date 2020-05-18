import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import format_sec, get_savepath, get_sourcecode, sin2

from external import KeysightN5173B, VaunixLMS, AgilentE8247C

sys.path.append(os.path.join("..", "server"))
import simpleq

# cavity drive: readout
brick_freq = 5_629_496_000.0  # Hz
# brick_pwr = 7.5  # dBm
brick_pwr = 22.0  # dBm
readout_freq = 400e6  # Hz
readout_amp = 4 * 0.03125  # FS
readout_phaseI = 0.
readout_phaseQ = np.pi / 2  # high sideband
readout_portI = 1
readout_portQ = 2

# qubit drive: control
gen_freq = 4_049_390_300.
gen_freq += 47455.  # from ramsey_lo_20191016_121306.npz
gen_pwr = 18.9  # dBm
control_freq = 300e6  # Hz
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

# sample
sample_portI = 1
sample_portQ = 2

# Rabi experiment
num_averages = 10_000
control_length = 320e-9
readout_length = 900e-9
sample_length = 1024e-9
nr_amps = 128
control_amp_array = np.linspace(0.0, 1.0, nr_amps)  # FS
rabi_readout_delay = 0.  # delay between control and readout pulses
rabi_decay = 500e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

# Program local oscillators
# with VaunixLMS() as brick:
with AgilentE8247C() as brick:
    brick.set_frequency(brick_freq)
    brick.set_power(brick_pwr)
    # brick.set_ext_ref(1)
    brick.set_output(1)
with KeysightN5173B() as source:
    source.set_frequency(gen_freq)
    source.set_power(gen_pwr)
    source.set_output(1)

# Instantiate interface class
with simpleq.SimpleQ() as q:
    # *** Set parameters ***
    # Set frequencies
    q.setup_freq_lut(readout_portI, readout_freq, [readout_phaseI], 1)
    q.setup_freq_lut(readout_portQ, readout_freq, [readout_phaseQ], 1)
    q.setup_freq_lut(control_portI, control_freq, [control_phaseI], 1)
    q.setup_freq_lut(control_portQ, control_freq, [control_phaseQ], 1)
    # Set amplitudes
    q.setup_scale_lut(readout_portI, readout_amp, 1)
    q.setup_scale_lut(readout_portQ, readout_amp, 1)
    q.setup_scale_lut(control_portI, control_amp_array, 1)
    q.setup_scale_lut(control_portQ, control_amp_array, 1)
    # Set pulses
    readout_pulseI = q.setup_long_drive(readout_portI, readout_length, use_scale=True)
    readout_pulseQ = q.setup_long_drive(readout_portQ, readout_length, use_scale=True)
    control_ns = int(round(control_length * q.sampling_freq))
    control_shape = sin2(control_ns)
    # control_shape = np.ones(control_ns)
    control_pulseI = q.setup_template(control_portI, control_shape, envelope=True, use_scale=True)
    control_pulseQ = q.setup_template(control_portQ, control_shape, envelope=True, use_scale=True)
    # Set sampling
    q.set_store_duration(sample_length)
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    # Control pulse
    q.output_carrier(T, control_length, [control_portI, control_portQ])
    q.output_pulse(T, [control_pulseI, control_pulseQ])
    # Readout pulse
    T += control_length + rabi_readout_delay
    q.output_pulse(T, [readout_pulseI, readout_pulseQ])
    # Sample
    q.store(T + readout_sample_delay)
    # Move to next Rabi amplitude
    T += readout_length
    q.next_scale(T, [control_portI, control_portQ])
    # Wait for decay
    T += rabi_decay

    expected_runtime = (T - T0) * nr_amps * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    t_array, result = q.perform_measurement(T,
                                            nr_amps,
                                            num_averages,
                                            print_time=True)

# *** Plot ***
fig, ax = plt.subplots(tight_layout=True)
# ax.plot(result[0, 0])
# ax.plot(result[0, 1])
ax.plot(1e9 * t_array, result[0, 0])
ax.plot(1e9 * t_array, result[0, 1])
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Voltage [FS]")
fig.show()

amps_i = np.zeros(nr_amps)
amps_q = np.zeros(nr_amps)
phases_i = np.zeros(nr_amps)
phases_q = np.zeros(nr_amps)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ii in range(nr_amps):
    data_i = result[ii, 0, idx_start:idx_stop]
    data_q = result[ii, 1, idx_start:idx_stop]
    fft_i = np.fft.rfft(data_i)
    fft_q = np.fft.rfft(data_q)
    amps_i[ii] = np.abs(fft_i[kk])
    amps_q[ii] = np.abs(fft_q[kk])
    phases_i[ii] = np.angle(fft_i[kk])
    phases_q[ii] = np.angle(fft_q[kk])

fig2, ax2 = plt.subplots(2, 1, tight_layout=True)
ax21, ax22 = ax2
ax21.plot(amps_i)
ax21.plot(amps_q)
ax22.plot(phases_i)
ax22.plot(phases_q)
fig2.show()

# *** Save ***
save_path = get_savepath(__file__)
sourcecode = get_sourcecode(__file__)
np.savez(
    save_path,
    num_averages=num_averages,
    control_freq=control_freq,
    readout_freq=readout_freq,
    readout_length=readout_length,
    readout_amp=readout_amp,
    control_amp_array=control_amp_array,
    sample_length=sample_length,
    control_length=control_length,
    rabi_readout_delay=rabi_readout_delay,
    rabi_decay=rabi_decay,
    readout_sample_delay=readout_sample_delay,
    brick_freq=brick_freq,
    brick_pwr=brick_pwr,
    gen_freq=gen_freq,
    gen_pwr=gen_pwr,
    result=result,
    sourcecode=sourcecode,
)
