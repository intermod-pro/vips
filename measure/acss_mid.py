import os
import sys

import numpy as np

from utils import format_sec, get_savepath, get_sourcecode, sin2

from external import KeysightN5173B, VaunixLMS

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
_gen_freq = 4_049_390_300.
_gen_freq += 47455.  # from ramsey_lo_20191016_121306.npz
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
control_length = 100e-9  # s, pi/2 pulse
control_amp = 0.5  # FS, pi/2 pulse

# sample
sample_portI = 1
sample_portQ = 2

# AC-Stark shift experiment
num_averages = 100_000  # 160_000
readout_length = 900e-9  # s
sample_length = 1024e-9
nr_delays = 128  # 64
dt_delays = 0.12e-6  # 0.25e-6  # s
wait_decay = 250e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

cavity_ringup = 2e-6

amp_array = np.sqrt(np.linspace(0, 0.023**2, 8))
# amp_array = np.array([0., 0.05])
# amp_array = np.array([0.])
nr_amps = len(amp_array)

detuning = -400e3
gen_freq = _gen_freq + detuning

# Program local oscillators
with VaunixLMS() as brick:
    brick.set_frequency(brick_freq)
    brick.set_power(brick_pwr)
    brick.set_ext_ref(1)
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
    q.setup_scale_lut(readout_portI, amp_array, 1)
    q.setup_scale_lut(readout_portQ, amp_array, 1)
    q.setup_scale_lut(control_portI, control_amp, 1)
    q.setup_scale_lut(control_portQ, control_amp, 1)
    # Set pulses
    readout_pulseI = q.setup_long_drive(readout_portI, readout_length, template_amp=readout_amp)
    readout_pulseQ = q.setup_long_drive(readout_portQ, readout_length, template_amp=readout_amp)
    ringup_pulseI = q.setup_long_drive(readout_portI, cavity_ringup, use_scale=True)
    ringup_pulseQ = q.setup_long_drive(readout_portQ, cavity_ringup, use_scale=True)
    control_ns = int(round(control_length * q.sampling_freq))
    if control_shape == "sin2":
        _template = sin2(control_ns)
    elif control_shape == "square":
        _template = np.ones(control_ns)
    else:
        raise NotImplementedError
    control_p_pulseI = q.setup_template(control_portI, _template, envelope=True, use_scale=True)
    control_p_pulseQ = q.setup_template(control_portQ, _template, envelope=True, use_scale=True)
    control_m_pulseI = q.setup_template(control_portI, -_template, envelope=True, use_scale=True)
    control_m_pulseQ = q.setup_template(control_portQ, -_template, envelope=True, use_scale=True)
    # Set sampling
    q.set_store_duration(sample_length)
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    for ii in range(nr_delays):
        for jj in range(2):
            # Fill up cavity
            _ringup = cavity_ringup + 2 * control_length + ii * dt_delays
            q._update_duration(ringup_pulseI, _ringup)
            q._update_duration(ringup_pulseQ, _ringup)
            q.output_pulse(T, [ringup_pulseI, ringup_pulseQ])
            # Control pulse 1
            T += cavity_ringup
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
    q.next_scale(T, [readout_portI, readout_portQ])
    T += wait_decay

    expected_runtime = (T - T0) * nr_amps * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    t_array, result = q.perform_measurement(T,
                                            nr_amps,
                                            num_averages,
                                            print_time=True)

assert result.shape == (nr_amps * 2 * nr_delays, 2, 4096)
result.shape = (nr_amps, nr_delays, 2, 2, -1)

# *** Save ***
save_path = get_savepath(__file__)
sourcecode = get_sourcecode(__file__)
np.savez(
    save_path,
    num_averages=num_averages,
    control_freq=control_freq,
    readout_freq=readout_freq,
    readout_length=readout_length,
    control_length=control_length,
    control_shape=control_shape,
    readout_amp=readout_amp,
    control_amp=control_amp,
    sample_length=sample_length,
    nr_delays=nr_delays,
    dt_delays=dt_delays,
    wait_decay=wait_decay,
    cavity_ringup=cavity_ringup,
    readout_sample_delay=readout_sample_delay,
    amp_array=amp_array,
    qubit_freq=_gen_freq,
    detuning=detuning,
    brick_freq=brick_freq,
    brick_pwr=brick_pwr,
    gen_freq=gen_freq,
    gen_pwr=gen_pwr,
    result=result,
    sourcecode=sourcecode,
)
