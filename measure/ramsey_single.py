import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import format_sec, get_savepath, get_sourcecode, sin2

sys.path.append(os.path.join("..", "server"))
import simpleq

sys.path.append('C:\\IMP Sessions and Settings\\settings\\startup_scripts')
from Periphery import Periphery

# cavity drive: readout
brick_freq = 5_629_496_000.0  # Hz
brick_pwr = 7.5  # dBm
readout_freq = 400e6  # Hz
readout_amp = 4 * 0.03125  # FS
readout_phaseI = 0.
readout_phaseQ = np.pi / 2  # high sideband
readout_portI = 1
readout_portQ = 2

# qubit drive: control
gen_freq = 4_049_390_300.

# !!!
gen_freq -= 250e3

gen_pwr = 19.0  # dBm
control_freq = 300e6  # Hz
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

control_shape = "square"
control_length = 44e-9  # s, pi/2 pulse
control_amp = 1.0  # FS, pi/2 pulse

# control_shape = "sin2"
# control_length = 300e-9  # s, pi/2 pulse
# control_amp = 0.200  # FS, pi/2 pulse

# sample
sample_portI = 1
sample_portQ = 2

# Ramsey experiment
num_averages = 10_000
readout_length = 900e-9  # s
sample_length = 1024e-9
nr_delays = 128
dt_delays = 0.2e-6  # s
wait_decay = 500e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

# Program local oscillators
p = Periphery()
Brick = Periphery.create_pinstrument(p, 'Brick', 'Vaunix_lab_brick', '0')
Brick.on()
Brick.set_power(brick_pwr)
Brick.set_external_reference()
Brick.set_frequency(brick_freq)
Source = Periphery.create_pinstrument(p, 'Source', 'Agilent_E8247C',
                                      '192.168.18.104')
Source.on()
Source.set_frequency(gen_freq / 1e9)
Source.set_power(gen_pwr)

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
    readout_pulseI = q.setup_continuous_drive(readout_portI, readout_length)
    readout_pulseQ = q.setup_continuous_drive(readout_portQ, readout_length)
    control_ns = int(round(control_length * q.sampling_freq))
    if control_shape == "sin2":
        _template = sin2(control_ns)
    elif control_shape == "square":
        _template = np.ones(control_ns)
    else:
        raise NotImplementedError
    control_pulseI = q.setup_template(control_portI, _template, envelope=True)
    control_pulseQ = q.setup_template(control_portQ, _template, envelope=True)
    # Set sampling
    q.set_store_duration(sample_length)
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    for ii in range(nr_delays):
        # Control pulse 1
        q.output_pulse(T, [control_pulseI, control_pulseQ])
        # Control pulse 2
        T += control_length + ii * dt_delays
        q.output_pulse(T, [control_pulseI, control_pulseQ])
        # Readout pulse
        T += control_length
        q.output_pulse(T, [readout_pulseI, readout_pulseQ])
        # Sample
        q.store(T + readout_sample_delay)
        # Wait for decay
        T += wait_decay

    expected_runtime = (T - T0) * 1 * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    t_array, result = q.perform_measurement(T, 1, num_averages, print_time=True)

Source._visainstrument.close()

# *** Plot ***
fig, ax = plt.subplots(tight_layout=True)
# ax.plot(result[0, 0])
# ax.plot(result[0, 1])
ax.plot(1e9 * t_array, result[0, 0])
ax.plot(1e9 * t_array, result[0, 1])
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Voltage [FS]")
fig.show()

amps_i = np.zeros(nr_delays)
amps_q = np.zeros(nr_delays)
phases_i = np.zeros(nr_delays)
phases_q = np.zeros(nr_delays)
idx_start = 2200
idx_stop = 3400
idx_span = idx_stop - idx_start
kk = int(round(readout_freq * (idx_span / 4e9)))

for ii in range(nr_delays):
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
