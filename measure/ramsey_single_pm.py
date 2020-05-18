import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from external import KeysightN5173B, AgilentE8247C
from utils import format_sec, get_savepath, get_sourcecode, sin2

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

control_shape = "square"
control_length = 20e-9  # s, pi/2 pulse
control_amp = 1.0  # FS, pi/2 pulse

# control_shape = "sin2"
# control_length = 38e-9  # s, pi/2 pulse
# control_amp = 1.0  # FS, pi/2 pulse

# sample
sample_portI = 1
sample_portQ = 2

# Ramsey experiment
num_averages = 10_000
readout_length = 900e-9  # s
sample_length = 1000e-9
detuning = -10e6
nr_delays = 128
dt_delays = 4e-9  # s
wait_decay = 250e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

gen_freq += detuning

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
    q.setup_freq_lut(readout_portI, readout_freq, readout_phaseI, 1)
    q.setup_freq_lut(readout_portQ, readout_freq, readout_phaseQ, 1)
    q.setup_freq_lut(control_portI, control_freq, control_phaseI, 1)
    q.setup_freq_lut(control_portQ, control_freq, control_phaseQ, 1)
    # Set amplitudes
    q.setup_scale_lut(readout_portI, readout_amp, 1)
    q.setup_scale_lut(readout_portQ, readout_amp, 1)
    q.setup_scale_lut(control_portI, control_amp, 1)
    q.setup_scale_lut(control_portQ, control_amp, 1)
    # Set pulses
    readout_pulseI = q.setup_long_drive(
        readout_portI,
        readout_length,
        use_scale=True,
    )
    readout_pulseQ = q.setup_long_drive(
        readout_portQ,
        readout_length,
        use_scale=True,
    )
    control_ns = int(round(control_length * q.sampling_freq))
    if control_shape == "sin2":
        _template = sin2(control_ns)
    elif control_shape == "square":
        _template = np.ones(control_ns)
    else:
        raise NotImplementedError
    control_p_pulseI = q.setup_template(
        control_portI,
        _template,
        envelope=True,
        use_scale=True,
    )
    control_p_pulseQ = q.setup_template(
        control_portQ,
        _template,
        envelope=True,
        use_scale=True,
    )
    control_m_pulseI = q.setup_template(
        control_portI,
        -_template,
        envelope=True,
        use_scale=True,
    )
    control_m_pulseQ = q.setup_template(
        control_portQ,
        -_template,
        envelope=True,
        use_scale=True,
    )
    # Set sampling
    q.set_store_duration(sample_length)
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    for ii in range(nr_delays):
        for jj in range(2):
            # Start carrier
            q.output_carrier(T, 2 * control_length + ii * dt_delays, [control_portI, control_portQ])
            # Control pulse 1
            q.output_pulse(T, [control_p_pulseI, control_p_pulseQ])
            # Control pulse 2
            T += control_length + ii * dt_delays
            if jj:
                q.output_pulse(T, [control_m_pulseI, control_m_pulseQ])
            else:
                q.output_pulse(T, [control_p_pulseI, control_p_pulseQ])
            # Readout pulse
            T += control_length
            q.output_pulse(T, [readout_pulseI, readout_pulseQ])
            # Sample
            q.store(T + readout_sample_delay)
            # Wait for decay
            T += wait_decay

    expected_runtime = (T - T0) * 1 * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    t_array, result = q.perform_measurement(T,
                                            1,
                                            num_averages,
                                            print_time=True)

result.shape = (nr_delays, 2, 2, -1)

# *** Plot ***
fig, ax = plt.subplots(tight_layout=True)
# ax.plot(result[0, 0])
# ax.plot(result[0, 1])
ax.plot(1e9 * t_array, result[0, 0, 0])
ax.plot(1e9 * t_array, result[0, 0, 1])
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Voltage [FS]")
fig.show()

amps_i = np.zeros((nr_delays, 2))
amps_q = np.zeros((nr_delays, 2))
phases_i = np.zeros((nr_delays, 2))
phases_q = np.zeros((nr_delays, 2))
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ii in range(nr_delays):
    for jj in range(2):
        data_i = result[ii, jj, 0, idx_start:idx_stop]
        data_q = result[ii, jj, 1, idx_start:idx_stop]
        fft_i = np.fft.rfft(data_i)
        fft_q = np.fft.rfft(data_q)
        amps_i[ii, jj] = np.abs(fft_i[kk])
        amps_q[ii, jj] = np.abs(fft_q[kk])
        phases_i[ii, jj] = np.angle(fft_i[kk])
        phases_q[ii, jj] = np.angle(fft_q[kk])

fig2, ax2 = plt.subplots(2, 1, tight_layout=True)
ax21, ax22 = ax2
ax21.plot(amps_i[:, 0])
ax21.plot(amps_q[:, 0])
ax21.plot(amps_i[:, 1])
ax21.plot(amps_q[:, 1])
ax22.plot(phases_i[:, 0])
ax22.plot(phases_q[:, 0])
ax22.plot(phases_i[:, 1])
ax22.plot(phases_q[:, 1])
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
    control_amp=control_amp,
    sample_length=sample_length,
    control_length=control_length,
    control_shape=control_shape,
    wait_decay=wait_decay,
    nr_delays=nr_delays,
    dt_delays=dt_delays,
    readout_sample_delay=readout_sample_delay,
    brick_freq=brick_freq,
    brick_pwr=brick_pwr,
    gen_freq=gen_freq,
    gen_pwr=gen_pwr,
    result=result,
    sourcecode=sourcecode,
)
