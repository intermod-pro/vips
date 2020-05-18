import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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
# readout_amp = 4 * 0.03125  # FS
readout_amp = 2.33 * 4 * 0.03125  # FS
# readout_amp = 1.0  # FS
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
# control_shape = "square"
# control_length = 40e-9  # s, pi pulse
# control_amp = 1.0  # FS, pi pulse
control_shape = "sin2"
control_length = 80e-9  # s, pi pulse
control_amp = 1.0  # FS, pi pulse

# sample
sample_portI = 1
sample_portQ = 2

# double-pulse experiment
num_averages = 4_000_000
readout_length = int(round(1 / bw / factor / 2e-9)) * 2e-9
sample_length = 1e-6
wait_decay = 500e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

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
    q.setup_freq_lut(control_portI, control_freq, control_phaseI, 1)
    q.setup_freq_lut(control_portQ, control_freq, control_phaseQ, 1)
    # Set amplitudes
    q.setup_scale_lut(readout_portI, 1.0, 1)
    q.setup_scale_lut(readout_portQ, 1.0, 1)
    q.setup_scale_lut(control_portI, control_amp, 1)
    q.setup_scale_lut(control_portQ, control_amp, 1)
    # Set pulses
    readout_ns = int(round(readout_length * q.sampling_freq))
    _base_template = sinP(1, readout_ns)
    # Relative amplitudes to keep same peak photon number
    # (from classical simulations)
    template_single = np.concatenate((_base_template, np.zeros(readout_ns)))
    template_double = 1.00 * np.concatenate((_base_template, -_base_template))
    # template_double = np.concatenate((_base_template, -_base_template))
    template_long = 0.570 * sinP(1, 2 * readout_ns)
    # template_long = sinP(1, 2 * readout_ns)
    template_cool = 3.43 * cool(2 * readout_ns)
    # template_cool = cool(2 * readout_ns)
    readout_pulse_singleI = q.setup_template(
        readout_portI,
        readout_amp * template_single,
        envelope=True,
    )
    readout_pulse_singleQ = q.setup_template(
        readout_portQ,
        readout_amp * template_single,
        envelope=True,
    )
    readout_pulse_doubleI = q.setup_template(
        readout_portI,
        readout_amp * template_double,
        envelope=True,
    )
    readout_pulse_doubleQ = q.setup_template(
        readout_portQ,
        readout_amp * template_double,
        envelope=True,
    )
    readout_pulse_longI = q.setup_template(
        readout_portI,
        readout_amp * template_long,
        envelope=True,
    )
    readout_pulse_longQ = q.setup_template(
        readout_portQ,
        readout_amp * template_long,
        envelope=True,
    )
    readout_pulse_coolI = q.setup_template(
        readout_portI,
        readout_amp * template_cool,
        envelope=True,
    )
    readout_pulse_coolQ = q.setup_template(
        readout_portQ,
        readout_amp * template_cool,
        envelope=True,
    )
    control_ns = int(round(control_length * q.sampling_freq))
    if control_shape == "sin2":
        template_pi = sin2(control_ns)
    elif control_shape == "square":
        template_pi = np.ones(control_ns)
    else:
        raise NotImplementedError
    control_pulseI = q.setup_template(
        control_portI,
        template_pi,
        envelope=True,
        use_scale=True,
    )
    control_pulseQ = q.setup_template(
        control_portQ,
        template_pi,
        envelope=True,
        use_scale=True,
    )
    # Set sampling
    q.set_store_duration(sample_length)
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    for jj in range(2):
        for ii in range(4):
            if jj == 0:
                # Ground state, no pulse
                pass
            elif jj == 1:
                # Excited state, pi pulse
                q.output_carrier(T, control_length, [control_portI, control_portQ])
                q.output_pulse(T, [control_pulseI, control_pulseQ])
            else:
                raise NotImplementedError
            T += control_length

            q.output_carrier(T, 2 * readout_length, [readout_portI, readout_portQ])
            if ii == 0:
                # Single pulse
                q.output_pulse(T,
                               [readout_pulse_singleI, readout_pulse_singleQ])
            elif ii == 1:
                # Double pulse
                q.output_pulse(T,
                               [readout_pulse_doubleI, readout_pulse_doubleQ])
            elif ii == 2:
                # Long pulse
                q.output_pulse(T, [readout_pulse_longI, readout_pulse_longQ])
            elif ii == 3:
                # Cool pulse
                q.output_pulse(T, [readout_pulse_coolI, readout_pulse_coolQ])
            else:
                raise NotImplementedError
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

template_arr = np.array([
    template_single,
    template_double,
    template_long,
    template_cool,
])
nr_templates = len(template_arr)
nr_samples = int(round(sample_length * q.sampling_freq))
result.shape = (2, nr_templates, 2, nr_samples)

# *** Plot ***
fig1, ax1 = plt.subplots(3, 1, sharex=True, sharey=True, tight_layout=True)
ax11, ax12, ax13 = ax1

ax11.plot(1e9 * t_array, result[0, 0, 0])
ax11.plot(1e9 * t_array, result[1, 0, 0])
ax12.plot(1e9 * t_array, result[0, 1, 0])
ax12.plot(1e9 * t_array, result[1, 1, 0])
ax13.plot(1e9 * t_array, result[0, 2, 0])
ax13.plot(1e9 * t_array, result[1, 2, 0])

ax1[-1].set_xlabel("Time [ns]")
for _ax in ax1:
    _ax.set_ylabel("Voltage [FS]")
fig1.show()

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
    readout_amp=readout_amp,
    control_amp=control_amp,
    template_arr=template_arr,
    template_pi=template_pi,
    sample_length=sample_length,
    wait_decay=wait_decay,
    readout_sample_delay=readout_sample_delay,
    brick_freq=brick_freq,
    brick_pwr=brick_pwr,
    gen_freq=gen_freq,
    gen_pwr=gen_pwr,
    result=result,
    sourcecode=sourcecode,
)
