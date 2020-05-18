import os
import sys

import numpy as np

from measure.utils import format_sec, get_savepath, get_sourcecode

sys.path.append(os.path.join("..", "server"))
import server.simpleq as simpleq

longString = ''

def addLine(longString, string):
    return longString + string + '\n'


def print_lines(longString):
    with open("Rabi_q_calls", 'w') as f:
        f.write(longString + '\n')

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
control_amp = 1.0  # FS
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

# sample
sample_portI = 1
sample_portQ = 2

# Rabi experiment
num_averages = 10_000
readout_length = 900e-9
sample_length = 1024e-9
rabi_n = 128
rabi_dt = 2e-9
rabi_readout_delay = 0.  # delay between control and readout pulses
rabi_decay = 500e-6  # delay between repetitions
readout_sample_delay = 200e-9  # delay between readout pulse and sample window

# detuning = -10e6
# gen_freq += detuning

# Instantiate interface class
with simpleq.SimpleQ(dry_run=True) as q:
    # *** Set parameters ***
    # Set frequencies
    longString = addLine(longString, f'q.setup_freq_lut({readout_portI}, {readout_freq}, {readout_phaseI}, 1)')
    q.setup_freq_lut(readout_portI, readout_freq, readout_phaseI, 1)
    longString = addLine(longString, f'q.setup_freq_lut({readout_portQ}, {readout_freq}, {readout_phaseQ}, 1)')
    q.setup_freq_lut(readout_portQ, readout_freq, readout_phaseQ, 1)
    longString = addLine(longString, f'q.setup_freq_lut({control_portI}, {control_freq}, {control_phaseI}, 1)')
    q.setup_freq_lut(control_portI, control_freq, control_phaseI, 1)
    longString = addLine(longString, f'q.setup_freq_lut({control_portQ}, {control_freq}, {control_phaseQ}, 1)')
    q.setup_freq_lut(control_portQ, control_freq, control_phaseQ, 1)
    # Set amplitudes
    longString = addLine(longString, f'q.setup_scale_lut({readout_portI}, {readout_amp}, 1)')
    q.setup_scale_lut(readout_portI, readout_amp, 1)
    longString = addLine(longString, f'q.setup_scale_lut({readout_portQ}, {readout_amp}, 1)')
    q.setup_scale_lut(readout_portQ, readout_amp, 1)
    longString = addLine(longString, f'q.setup_scale_lut({control_portI}, {control_amp}, 1)')
    q.setup_scale_lut(control_portI, control_amp, 1)
    longString = addLine(longString, f'q.setup_scale_lut({control_portQ}, {control_amp}, 1)')
    q.setup_scale_lut(control_portQ, control_amp, 1)
    # Set pulses
    longString = addLine(longString, f'q.setup_long_drive({readout_portI}, {readout_length}, use_scale={True})')
    readout_pulseI = q.setup_long_drive(readout_portI, readout_length, use_scale=True)
    longString = addLine(longString, f'q.setup_long_drive({readout_portQ}, {readout_length}, use_scale={True})')
    readout_pulseQ = q.setup_long_drive(readout_portQ, readout_length, use_scale=True)
    longString = addLine(longString, f'q.setup_long_drive({control_portI}, {rabi_dt}, use_scale={True})')
    control_pulseI = q.setup_long_drive(control_portI, rabi_dt, use_scale=True)
    longString = addLine(longString, f'q.setup_long_drive({control_portQ}, {rabi_dt}, use_scale={True})')
    control_pulseQ = q.setup_long_drive(control_portQ, rabi_dt, use_scale=True)
    # Set sampling
    longString = addLine(longString, f'q.set_store_duration({sample_length})')
    q.set_store_duration(sample_length)
    longString = addLine(longString, f'q.set_store_ports({[sample_portI, sample_portQ]})')
    q.set_store_ports([sample_portI, sample_portQ])

    # *** Program pulse sequence ***
    T0 = 2e-6  # s, start from some time
    T = T0
    for ii in range(rabi_n):
        longString = addLine(longString, str(ii))
        #if ii:
        # Rabi pulse
        rabi_length = ii * rabi_dt
        longString = addLine(longString, f'q._update_duration({control_pulseI}, {rabi_length})')
        q._update_duration(control_pulseI, rabi_length)
        longString = addLine(longString, f'q._update_duration({control_pulseQ}, {rabi_length})')
        q._update_duration(control_pulseQ, rabi_length)
        longString = addLine(longString, f'q._output_pulse({T}, {[control_pulseI, control_pulseQ]})')
        q.output_pulse(T, [control_pulseI, control_pulseQ])
        # Readout pulse
        T += ii * rabi_dt + rabi_readout_delay
        longString = addLine(longString, f'q._output_pulse({T}, {[readout_pulseI, readout_pulseQ]})')
        q.output_pulse(T, [readout_pulseI, readout_pulseQ])
        # Sample
        longString = addLine(longString, f'q.store({T + readout_sample_delay})')
        q.store(T + readout_sample_delay)
        # Move to next Rabi length
        T += rabi_decay
    print_lines(longString)
    expected_runtime = (T - T0) * num_averages  # s
    print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

    # with open('Rabi_Seq_output', 'w') as f:
    #     seq = q.seq
    #     for line in seq:
    #         parts = str(line).split(',', 1)
    #         f.write(parts[1] + '\n')
    # t_array, result = q.perform_measurement(T,
    #                                         1,
    #                                         num_averages,
    #                                         print_time=True)
