import os
import sys

import numpy as np

from utils import format_sec, get_savepath, get_sourcecode
from utils import cool, sin2, sinP

from external import KeysightN5173B, AgilentE8247C

sys.path.append(os.path.join("..", "server"))
import simpleq

# system
fc = 6_029_359_800.
wc = 2. * np.pi * fc
bw = 904e3
kappa = 2. * np.pi * bw
factor = 10

# cavity drive: readout
readout_freq = 400e6  # Hz
# fast_readout_amp = 2.33 * 4 * 0.03125  # FS
fast_readout_amp = 1.0  # FS
normal_readout_amp = 4 * 0.03125  # FS
readout_phaseI = 0.
readout_phaseQ = np.pi / 2  # high sideband
readout_portI = 1
readout_portQ = 2
brick_freq = fc - readout_freq
brick_pwr = 22.0  # dBm

# qubit drive: control
gen_freq = 4_049_390_300.
gen_freq += 47455.  # from ramsey_lo_20191016_121306.npz
gen_pwr = 18.9  # dBm
control_freq = 300e6  # Hz
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

pi_pulse_shape = "square"
pi_pulse_length = 40e-9  # s, pi pulse
pi_pulse_amp = 1.0  # FS, pi pulse

pi2_pulse_shape = "square"
pi2_pulse_length = 20e-9  # s, pi/2 pulse
pi2_pulse_amp = 1.0  # FS, pi/2 pulse

# sample
sample_portI = 1
sample_portQ = 2

# Ramsey
detuning = -10e6
nr_delays = 127
dt_delays = 6e-9  # s

# double-pulse experiment
num_averages = 100_000
fast_readout_length = int(round(1 / bw / factor / 2e-9)) * 2e-9
normal_readout_length = 900e-9
sample_length = 1e-6
wait_decay = 250e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window
buffer_time = 900e-9  # delay between fast and normal readout to ensure photons from former are gone

# which_pulse = "single"
# which_pulse = "double"
# which_pulse = "long"
# which_pulse = "cool"
# which_pulse = "none"

which_pulse = sys.argv[1]
print()
print("*******")
print(which_pulse)
print("*******")
print()

# gen_freq += detuning
# do detuning on IF

# Program local oscillators
with AgilentE8247C() as brick:
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
    q.setup_freq_lut(control_portI, [control_freq, control_freq - detuning], control_phaseI * np.ones(2), 1)
    q.setup_freq_lut(control_portQ, [control_freq, control_freq - detuning], control_phaseQ * np.ones(2), 1)
    # Set amplitudes
    q.setup_scale_lut(readout_portI, 1.0, 1)
    q.setup_scale_lut(readout_portQ, 1.0, 1)
    q.setup_scale_lut(control_portI, [0.0, 1.0], 1)
    q.setup_scale_lut(control_portQ, [0.0, 1.0], 1)

    # Set pulses
    # Fast readout pulses
    fast_readout_ns = int(round(fast_readout_length * q.sampling_freq))
    # Relative amplitudes to keep same peak photon number
    # (from classical simulations)
    if which_pulse == "single":
        _base_template = sinP(1, fast_readout_ns)
        template_fast = np.concatenate(
            (_base_template, np.zeros(fast_readout_ns)))
    elif which_pulse == "single -- right away":
        _base_template = sinP(1, fast_readout_ns)
        template_fast = _base_template
    elif which_pulse == "double":
        _base_template = sinP(1, fast_readout_ns)
        template_fast = 1.00 * np.concatenate(
            (_base_template, -_base_template))
    elif which_pulse == "long":
        # template_fast = 0.570 * sinP(1, 2 * fast_readout_ns)
        template_fast = sinP(1, 2 * fast_readout_ns)
    elif which_pulse == "cool":
        # template_fast = 3.43 * cool(2 * fast_readout_ns)
        template_fast = cool(2 * fast_readout_ns)
    elif which_pulse == "none":
        template_fast = np.zeros(2 * fast_readout_ns)
    else:
        raise NotImplementedError
    fast_readout_pulseI = q.setup_template(
        readout_portI,
        fast_readout_amp * template_fast,
        envelope=True,
    )
    fast_readout_pulseQ = q.setup_template(
        readout_portQ,
        fast_readout_amp * template_fast,
        envelope=True,
    )

    # Normal readout pulse
    normal_readout_pulseI = q.setup_long_drive(
        readout_portI,
        normal_readout_length,
        template_amp=normal_readout_amp,
    )
    normal_readout_pulseQ = q.setup_long_drive(
        readout_portQ,
        normal_readout_length,
        template_amp=normal_readout_amp,
    )

    # Control pulses
    # pi pulse
    pi_pulse_ns = int(round(pi_pulse_length * q.sampling_freq))
    if pi_pulse_shape == "sin2":
        template_pi = sin2(pi_pulse_ns)
    elif pi_pulse_shape == "square":
        template_pi = np.ones(pi_pulse_ns)
    else:
        raise NotImplementedError
    template_pi *= pi_pulse_amp
    pi_pulseI = q.setup_template(
        control_portI,
        template_pi,
        envelope=True,
        use_scale=True,
    )
    pi_pulseQ = q.setup_template(
        control_portQ,
        template_pi,
        envelope=True,
        use_scale=True,
    )
    # pi/2 pulse
    pi2_pulse_ns = int(round(pi2_pulse_length * q.sampling_freq))
    if pi2_pulse_shape == "sin2":
        template_pi2 = sin2(pi2_pulse_ns)
    elif pi2_pulse_shape == "square":
        template_pi2 = np.ones(pi2_pulse_ns)
    else:
        raise NotImplementedError
    template_pi2 *= pi2_pulse_amp
    pi2_pulseI = q.setup_template(
        control_portI,
        template_pi2,
        envelope=True,
    )
    pi2_pulseQ = q.setup_template(
        control_portQ,
        template_pi2,
        envelope=True,
    )
    # -pi/2
    mpi2_pulseI = q.setup_template(
        control_portI,
        -template_pi2,
        envelope=True,
    )
    mpi2_pulseQ = q.setup_template(
        control_portQ,
        -template_pi2,
        envelope=True,
    )
    # Set sampling
    q.set_store_duration(sample_length)
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    for jj in range(nr_delays):  # Ramsey
        for ii in range(2):  # +/- pi/2
            # ground or excited state
            # pi pulse controlled by scale
            q.output_carrier(T, pi_pulse_length, [control_portI, control_portQ])
            q.output_pulse(T, [pi_pulseI, pi_pulseQ])
            T += pi_pulse_length
            q.next_frequency(T, [control_portI, control_portQ])

            # Fast-readout pulse (double)
            if which_pulse == "single -- right away":
                q.output_carrier(T, fast_readout_length, [readout_portI, readout_portQ])
            else:
                q.output_carrier(T, 2 * fast_readout_length, [readout_portI, readout_portQ])
            q.output_pulse(T, [fast_readout_pulseI, fast_readout_pulseQ])
            if which_pulse == "single -- right away":
                T += fast_readout_length
            else:
                T += 2 * fast_readout_length

            # Ramsey sequence
            q.output_carrier(T, 2 * pi2_pulse_length + jj * dt_delays, [control_portI, control_portQ])
            # First pi/2 pulse
            q.output_pulse(T, [pi2_pulseI, pi2_pulseQ])
            # Second pi/2 pulse
            T += pi2_pulse_length + jj * dt_delays
            if ii:
                q.output_pulse(T, [mpi2_pulseI, mpi2_pulseQ])
            else:
                q.output_pulse(T, [pi2_pulseI, pi2_pulseQ])
            T += pi2_pulse_length
            q.next_frequency(T, [control_portI, control_portQ])

            # Normal-readout pulse
            T += buffer_time - (2 * pi2_pulse_length + jj * dt_delays)
            q.output_pulse(T, [normal_readout_pulseI, normal_readout_pulseQ])
            # Sample
            q.store(T + readout_sample_delay)
            # Wait for decay
            T += wait_decay
    q.next_scale(T, [control_portI, control_portQ])
    T += 2e-6

    expected_runtime = (T - T0) * 2 * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    t_array, result = q.perform_measurement(T,
                                            2,
                                            num_averages,
                                            print_time=True)

nr_samples = int(round(sample_length * q.sampling_freq))
result.shape = (2, nr_delays, 2, 2, nr_samples)


# *** Save ***
save_path = get_savepath(__file__)
sourcecode = get_sourcecode(__file__)
np.savez(
    save_path,
    num_averages=num_averages,
    control_freq=control_freq,
    readout_freq=readout_freq,
    fast_readout_length=fast_readout_length,
    normal_readout_length=normal_readout_length,
    pi_pulse_length=pi_pulse_length,
    pi2_pulse_length=pi2_pulse_length,
    fast_readout_amp=fast_readout_amp,
    normal_readout_amp=normal_readout_amp,
    pi_pulse_amp=pi_pulse_amp,
    pi2_pulse_amp=pi2_pulse_amp,
    which_pulse=which_pulse,
    template_fast=template_fast,
    template_pi=template_pi,
    template_pi2=template_pi2,
    sample_length=sample_length,
    wait_decay=wait_decay,
    readout_sample_delay=readout_sample_delay,
    brick_freq=brick_freq,
    brick_pwr=brick_pwr,
    gen_freq=gen_freq,
    gen_pwr=gen_pwr,
    detuning=detuning,
    nr_delays=nr_delays,
    dt_delays=dt_delays,
    result=result,
    sourcecode=sourcecode,
)
