import os
import sys

import numpy as np

from external import KeysightN5173B, VaunixLMS
from utils import get_savepath, get_sourcecode, sin2

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
gen_freq += 47455.  # from ramsey_lo_20191016_121306.npz
gen_pwr = 18.9  # dBm
control_freq = 300e6  # Hz
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

# control_shape = "square"
# control_length = 44e-9  # s, pi/2 pulse
# control_amp = 1.0  # FS, pi/2 pulse

control_shape = "sin2"
control_length = 196e-9  # s, pi/2 pulse
control_amp = 0.5  # FS, pi/2 pulse

# sample
sample_portI = 1
sample_portQ = 2

# Excited sweep experiment
num_averages = 10_000
# readout_length = 900e-9  # s
readout_length = 3e-6  # s
sample_length = 1e-6
wait_decay = 250e-6  # delay between repetitions
readout_sample_delay = 250e-9 + 1e-6  # delay between readout pulse and sample window
_freq_span = 10e6
nr_freqs = 512
detuning_array = np.linspace(-_freq_span / 2, _freq_span / 2, nr_freqs)

# Program local oscillators
with KeysightN5173B() as source:
    source.set_frequency(gen_freq)
    source.set_power(gen_pwr)
    source.set_output(1)

with VaunixLMS() as brick:
    brick.set_frequency(brick_freq)
    brick.set_power(brick_pwr)
    brick.set_ext_ref(1)
    brick.set_output(1)

    result = np.zeros((nr_freqs, 2, 2, int(round(sample_length * 4e9))))
    for kk, detuning in enumerate(detuning_array):
        print(kk)
        brick.set_frequency(brick_freq + detuning)
        # Instantiate interface class
        with simpleq.SimpleQ() as q:
            # *** Set parameters ***
            # Set frequencies
            q.setup_freq_lut(readout_portI, readout_freq, readout_phaseI, 1)
            q.setup_freq_lut(readout_portQ, readout_freq, readout_phaseQ, 1)
            q.setup_freq_lut(control_portI, control_freq, control_phaseI, 1)
            q.setup_freq_lut(control_portQ, control_freq, control_phaseQ, 1)
            # Set amplitudes
            q.setup_scale_lut(readout_portI, 1.0, 1)
            q.setup_scale_lut(readout_portQ, 1.0, 1)
            q.setup_scale_lut(control_portI, 1.0, 1)
            q.setup_scale_lut(control_portQ, 1.0, 1)
            # Set pulses
            readout_pulseI = q.setup_long_drive(readout_portI,
                                                readout_length,
                                                template_amp=readout_amp)
            readout_pulseQ = q.setup_long_drive(readout_portQ,
                                                readout_length,
                                                template_amp=readout_amp)
            control_ns = int(round(control_length * q.sampling_freq))
            if control_shape == "sin2":
                _template = sin2(control_ns)
            elif control_shape == "square":
                _template = np.ones(control_ns)
            else:
                raise NotImplementedError
            _template *= control_amp
            control_pulseI = q.setup_template(control_portI,
                                              _template,
                                              envelope=True)
            control_pulseQ = q.setup_template(control_portQ,
                                              _template,
                                              envelope=True)
            # Set sampling
            q.set_store_duration(sample_length)
            q.set_store_ports([sample_portI, sample_portQ])

            # *** Program pulse sequence ***
            T0 = 2e-6  # s, start from some time
            T = T0
            for ii in range(2):
                if ii:
                    # pi pulse
                    q.output_pulse(T, [control_pulseI, control_pulseQ])
                # Readout pulse
                T += control_length
                q.output_pulse(T, [readout_pulseI, readout_pulseQ])
                # Sample
                q.store(T + readout_sample_delay)
                # Wait for decay
                T += wait_decay
            q.next_frequency(T, [readout_portI, readout_portQ])
            T += wait_decay

            expected_runtime = (T - T0) * nr_freqs * num_averages  # s

            t_array, _result = q.perform_measurement(T,
                                                     1,
                                                     num_averages,
                                                     print_time=True)
            result[kk, :, :, :] = _result[:, :, :]

    brick.set_frequency(brick_freq)

# *** Save ***
save_path = get_savepath(__file__)
sourcecode = get_sourcecode(__file__)
np.savez(
    save_path,
    num_averages=num_averages,
    control_freq=control_freq,
    readout_freq=readout_freq,
    control_length=control_length,
    readout_length=readout_length,
    readout_amp=readout_amp,
    control_amp=control_amp,
    control_shape=control_shape,
    sample_length=sample_length,
    wait_decay=wait_decay,
    readout_sample_delay=readout_sample_delay,
    detuning_array=detuning_array,
    brick_freq=brick_freq,
    brick_pwr=brick_pwr,
    gen_freq=gen_freq,
    gen_pwr=gen_pwr,
    result=result,
    sourcecode=sourcecode,
)
