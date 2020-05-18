import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from utils import format_sec, get_savepath, get_sourcecode, sin2

sys.path.append(os.path.join("..", "server"))
import simpleq

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
gen_pwr = 19.0  # dBm
control_freq = 300e6  # Hz, the detuning is calculated from here
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

control_shape = "square"
control_length = 40e-9  # s, pi/2 pulse
control_amp = 1.0  # FS, pi/2 pulse

# control_shape = "sin2"
# control_length = 300e-9  # s, pi/2 pulse
# control_amp = 0.200  # FS, pi/2 pulse

# sample
sample_portI = 1
sample_portQ = 2

# Ramsey experiment
num_averages = 1_000
readout_length = 900e-9  # s
sample_length = 1024e-9
nr_delays = 38
dt_delays = 0.8e-6  # s
wait_decay = 500e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

nr_freqs = 32
detuning_array = np.linspace(-250e3, 250e3, nr_freqs)

nr_samples = int(round(sample_length * 4e9))
result = np.zeros((nr_freqs, nr_delays, 2, 2, nr_samples))

for ff, detuning in enumerate(detuning_array):
    # Instantiate interface class
    with simpleq.SimpleQ(dry_run=True) as q:
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
        control_p_pulseI = q.setup_template(control_portI, _template, envelope=True)
        control_p_pulseQ = q.setup_template(control_portQ, _template, envelope=True)
        control_m_pulseI = q.setup_template(control_portI, -_template, envelope=True)
        control_m_pulseQ = q.setup_template(control_portQ, -_template, envelope=True)
        # Set sampling
        q.set_store_duration(sample_length)
        q.set_store_ports([sample_portI, sample_portQ])

        # *** Program pulse sequence ***
        T0 = 2e-6  # s, start from some time
        T = T0
        for ii in range(nr_delays):
            for jj in range(2):
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

        expected_runtime = (T - T0) * nr_freqs * num_averages  # s
        print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

        assert False
        t_array, _result = q.perform_measurement(T, nr_freqs, num_averages, print_time=True)
        _result.shape = (nr_delays, 2, 2, -1)

        result[ff, :, :, :, :] = _result[:, :, :, :]
