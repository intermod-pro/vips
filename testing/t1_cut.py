import numpy as np

from measure.utils import format_sec, get_savepath, get_sourcecode, sin2

import server.simpleq as simpleq

# cavity drive: readout
brick_freq = 5_629_496_000.0  # Hz
brick_pwr = 7.5  # dBm
readout_freq = 400e6  # Hz
readout_length = 900e-9  # s
readout_amp = 4 * 0.03125  # FS
readout_phaseI = 0.
readout_phaseQ = np.pi / 2  # high sideband
readout_portI = 1
readout_portQ = 2

# qubit drive: control
gen_freq = 4_049_390_300.
gen_pwr = 19.0  # dBm
control_freq = 300e6  # Hz
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

# control_shape = "square"
# control_length = 82e-9  # s, pi pulse
# control_amp = 1.0  # FS, pi pulse

control_shape = "sin2"
control_length = 250e-9  # s, pi pulse
control_amp = 0.912  # FS, pi pulse

# sample
sample_portI = 1
sample_portQ = 2

# T1 experiment
num_averages = 100_000
sample_length = 1024e-9
nr_delays = 511
dt_delays = 200e-9  # s
wait_decay = 500e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

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
    readout_pulseI = q.setup_long_drive(readout_portI, readout_length)
    readout_pulseQ = q.setup_long_drive(readout_portQ, readout_length)
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
        # Control pulse
        q.output_pulse(T, [control_pulseI, control_pulseQ])
        # Readout pulse
        T += control_length + ii * dt_delays
        q.output_pulse(T, [readout_pulseI, readout_pulseQ])
        # Sample
        q.store(T + readout_sample_delay)
        # Wait for decay
        T += wait_decay

    expected_runtime = (T - T0) * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    with open('T1_Seq_output', 'w') as f:
        seq = q.seq
        for line in seq:
            f.write(str(line) + '\n')
    #t_array, result = q.perform_measurement(T, 1, num_averages, print_time=True)

