import os
# https://github.com/ContinuumIO/anaconda-issues/issues/905#issuecomment-232498034
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import os
import signal
import sys
import time

from matplotlib import _pylab_helpers
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import format_sec, get_savepath, get_sourcecode, untwist_downconversion, sin2

sys.path.append(os.path.join("..", "server"))
import simpleq

sys.path.append('C:\\IMP Sessions and Settings\\settings\\startup_scripts')
from Periphery import Periphery


def handler(signum, frame):
    global KEEP_GOING
    if KEEP_GOING:
        print("\n\n")
        print("Ctrl-C pressed!")
        print("Will finish this run and then stop.")
        print("Press Ctrl-C again to abort.")
        print("\n\n")
        KEEP_GOING = False
    else:
        raise KeyboardInterrupt


def func(t, offset, amplitude, T2, period, phase):
    frequency = 1 / period
    return offset + amplitude * np.exp(
        -t / T2) * np.cos(2. * np.pi * frequency * t + phase)


def fit_period(x, y):
    pkpk = np.max(y) - np.min(y)
    offset = np.min(y) + pkpk / 2
    amplitude = 0.5 * pkpk
    T2 = 0.5 * (np.max(x) - np.min(x))
    freqs = np.fft.rfftfreq(len(x), x[1] - x[0])
    fft = np.fft.rfft(y)
    frequency = freqs[1 + np.argmax(np.abs(fft[1:]))]
    period = 1 / frequency
    first = (y[0] - offset) / amplitude
    if first > 1.:
        first = 1.
    elif first < -1.:
        first = -1.
    phase = np.arccos(first)
    p0 = (
        offset,
        amplitude,
        T2,
        period,
        phase,
    )
    popt, pcov = curve_fit(func, x, y, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    print(perr)
    offset, amplitude, T2, period, phase = popt
    return popt, perr


def my_pause(interval=0.1):
    manager = _pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        # plt.show(block=False)
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)


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
# gen_freq = 4_049_307_100.0  # Hz
gen_freq = 4_049_390_300.
gen_pwr = 19.0  # dBm
control_freq = 300e6  # Hz
control_phaseI = 0.
control_phaseQ = -np.pi / 2  # low sideband
control_portI = 3
control_portQ = 4

# sample
sample_portI = 1
sample_portQ = 2

# Rabi experiment
num_averages = 5_000
control_length = 500e-9
readout_length = 900e-9
sample_length = 1024e-9
nr_amps = 64
control_amp_array = np.linspace(0.0, 1.0, nr_amps)  # FS
rabi_readout_delay = 0.  # delay between control and readout pulses
rabi_decay = 500e-6  # delay between repetitions
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


fig, ax = plt.subplots(2, 1, tight_layout=True)
ax1, ax2 = ax

time_arr = np.array([])
rel_time_arr = np.array([])
period_arr = np.array([])
line1, = ax1.plot(rel_time_arr, period_arr, '.')
ax1.set_ylabel(r"$2\pi$ amplitude [$\mathrm{FS}$]")
ax1.set_xlabel(r"Time since start [$\mathrm{s}$]")

my_signal = np.zeros_like(control_amp_array)
line2, = ax2.plot(control_amp_array, my_signal, label='data')
line2f, = ax2.plot(control_amp_array, my_signal, '--', label='fit')
ax2.set_xlabel(r"Pulse length $\Delta t$ [$\mathrm{\mu s}$]")
ax2.set_ylabel(r"$A$")

fig.show()
my_pause()

KEEP_GOING = True
signal.signal(signal.SIGINT, handler)
count = 0
struct_time_start = time.localtime()
while KEEP_GOING:
    print("\n\n")
    print("*** Run number {:d}".format(count + 1))
    count += 1

    print("Setup")
    # Instantiate interface class
    with simpleq.SimpleQ() as q:
        # *** Set parameters ***
        # Set frequencies
        q.setup_freq_lut(readout_portI, [readout_freq], [readout_phaseI], 1)
        q.setup_freq_lut(readout_portQ, [readout_freq], [readout_phaseQ], 1)
        q.setup_freq_lut(control_portI, [control_freq], [control_phaseI], 1)
        q.setup_freq_lut(control_portQ, [control_freq], [control_phaseQ], 1)
        # Set amplitudes
        q.setup_scale_lut(readout_portI, [readout_amp], 1)
        q.setup_scale_lut(readout_portQ, [readout_amp], 1)
        q.setup_scale_lut(control_portI, control_amp_array, 1)
        q.setup_scale_lut(control_portQ, control_amp_array, 1)
        # Set pulses
        readout_pulseI = q.setup_continuous_drive(readout_portI, readout_length)
        readout_pulseQ = q.setup_continuous_drive(readout_portQ, readout_length)
        control_ns = int(round(control_length * q.sampling_freq))
        control_shape = sin2(control_ns)
        # control_shape = np.ones(control_ns)
        control_pulseI = q.setup_template(control_portI, control_shape, envelope=True)
        control_pulseQ = q.setup_template(control_portQ, control_shape, envelope=True)
        # Set sampling
        q.set_store_duration(sample_length)
        q.set_store_ports([sample_portI, sample_portQ])

        # *** Program pulse sequence ***
        T0 = 2e-6  # s, start from some time
        T = T0
        # Control pulse
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

        print("Measurig")
        t_array, result = q.perform_measurement(T,
                                                nr_amps,
                                                num_averages,
                                                print_time=True)

    print("Analyzing")

    I_port = np.zeros(nr_amps, dtype=np.complex128)
    Q_port = np.zeros(nr_amps, dtype=np.complex128)
    idx_start = 2200
    idx_stop = 3400
    idx_span = idx_stop - idx_start
    kk = int(round(readout_freq * (idx_span / 4e9)))

    for ii in range(nr_amps):
        data_i = result[ii, 0, idx_start:idx_stop]
        data_q = result[ii, 1, idx_start:idx_stop]
        fft_i = np.fft.rfft(data_i) / idx_span
        fft_q = np.fft.rfft(data_q) / idx_span
        I_port[ii] = fft_i[kk]
        Q_port[ii] = fft_q[kk]

    L_sideband, H_sideband = untwist_downconversion(I_port, Q_port)

    # my_signal = np.real(H_sideband)
    # my_signal = np.imag(H_sideband)
    my_signal = np.abs(H_sideband)
    # my_signal = np.unwrap(np.angle(H_sideband))

    popt, perr = fit_period(control_amp_array, my_signal)
    my_fit = func(control_amp_array, *popt)

    period = popt[3]
    print("Pi pulse amplitude: {}".format(period / 2))
    print("Pi/2 pulse amplitude: {}".format(period / 4))

    period_arr = np.concatenate((period_arr, [period]))
    time_arr = np.concatenate((time_arr, [time.time()]))
    rel_time_arr = time_arr - time_arr[0]

    line1.set_data(rel_time_arr, period_arr)

    line2.set_ydata(1e6 * my_signal)
    line2f.set_ydata(1e6 * my_fit)

    ax1.relim()
    ax1.autoscale()
    ax2.relim()
    ax2.autoscale()

    my_pause()

    print("Saving")
    # *** Save ***
    save_path = get_savepath(__file__, struct_time=struct_time_start)
    sourcecode = get_sourcecode(__file__)
    np.savez(
        save_path,
        num_averages=num_averages,
        control_frequency=control_freq,
        readout_frequency=readout_freq,
        readout_length=readout_length,
        readout_amplitude=readout_amp,
        control_amp_array=control_amp_array,
        control_length=control_length,
        sample_length=sample_length,
        rabi_readout_delay=rabi_readout_delay,
        rabi_decay=rabi_decay,
        readout_sample_delay=readout_sample_delay,
        brick_freq=brick_freq,
        brick_pwr=brick_pwr,
        gen_freq=gen_freq,
        gen_pwr=gen_pwr,
        time_arr=time_arr,
        period_arr=period_arr,
        my_signal=my_signal,
        my_fit=my_fit,
        sourcecode=sourcecode,
    )

Source._visainstrument.close()
print("\n\n")
print("Done")
input("___ Press Enter to close ___")
