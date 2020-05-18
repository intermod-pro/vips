"""
Definition of class SimpleQ which is an easy to use interface to the "RFZynq Quantum Firmware".
"""

import time

import numpy as np

from . import rflockin
from . import generate_quantum_stimuli as gs

MAX_LUT_ENTRIES = 512
MAX_TRIGGERS = 4096


def format_sec(s):
    """ Utility function to format a time interval in seconds
    into a more human-readable string.

    Args:
        s (float): time interval in seconds

    Returns:
        (str): time interval in the form "X h Y m Z.z s"

    Examples:
        >>> format_sec(12345.6)
        '3h 25m 45.6s'
    """
    if s < 1.:
        return "{:.1f}ms".format(s * 1e3)

    h = int(s // 3600)
    s -= h * 3600.

    m = int(s // 60)
    s -= m * 60

    if h:
        res = "{:d}h {:d}m {:.1f}s".format(h, m, s)
    elif m:
        res = "{:d}m {:.1f}s".format(m, s)
    else:
        res = "{:.1f}s".format(s)

    return res


class SimpleQ():
    """
    Class docstring
    """

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        if self.dry_run == False:
            self.rflockin = rflockin.rflockin()
        self.gqs = gs.generate_quantum_stimuli()
        self.used_templates = [0] * 8
        self.modulate = 0
        self.seq = []
        self.saves = []
        self.store_duration = None
        self.user_store_duration = None
        self.store_ports = None
        self.sampling_freq = 4e9

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """
        Close connection to hardware.
        """
        if self.dry_run == False:
            self.rflockin.close()

    def _time_to_index(self, t):
        return int(np.round(t / 2e-9))

    def setup_freq_lut(self, output_index, frequency_lut, phase_lut,
                       repeat_count):
        """
        Setup look-up table for frequency generator.
        """

        if output_index < 1 or output_index > 8:
            raise ValueError(
                'Invalid output, expected 1-8, got {}'.format(output_index))
        output_index -= 1

        frequency_lut = np.atleast_1d(frequency_lut).astype(np.float64)
        phase_lut = np.atleast_1d(phase_lut).astype(np.float64)

        if len(phase_lut) != len(frequency_lut):
            raise ValueError(
                'frequency_lut and phase_lut must have the same dimensions')
        if len(frequency_lut) > MAX_LUT_ENTRIES:
            raise ValueError(
                'frequency_lut can contain at most {:d} elements'.format(
                    MAX_LUT_ENTRIES))
        if np.any(frequency_lut > 2e9) or np.any(frequency_lut < 0):
            raise ValueError('Invalid frequency')

        increment_lut = np.round(2**40 / 500e6 * frequency_lut).astype(
            np.uint64)
        phase_lut = np.round(phase_lut / (2 * np.pi) * 2**40).astype(np.uint64)

        self.gqs.setup_freq_lut(output_index, increment_lut, phase_lut,
                                int(np.round(2**10 / repeat_count)))

    def setup_scale_lut(self, output_index, scale, repeat_count):
        """
        Setup look-up table for amplitude scale.
        """
        scale = np.atleast_1d(scale).astype(np.float64)
        if len(scale) > MAX_LUT_ENTRIES:
            raise ValueError('scale can contain at most {:d} elements'.format(
                MAX_LUT_ENTRIES))
        if np.any(np.abs(scale) > 1.0):
            raise ValueError('scale must be in [-1.0; +1.0]')
        scale_lut = np.int32(scale * 65536)
        self.gqs.setup_scale_lut(output_index - 1, scale_lut,
                                 int(np.round(2**10 / repeat_count)))

    def setup_long_drive(self, output_port, duration, template_amp=1.0, use_scale=False):
        """
        Setup waveform generator to output a template.
        """
        output_index = output_port - 1

        if self.used_templates[output_index] > 15:
            raise RuntimeError(
                'Not enough templates on output {}'.format(output_index))

        if np.abs(template_amp) > 1.0:
            raise ValueError("Scale must be in [-1.0; +1.0]")
        template = template_amp * np.ones(32)
        # scale from 1.0 full scale to adc units full scale
        template = np.int16(32767 * template)

        triginfo = []
        triginfo.append(
            ('carrier', output_index, self._time_to_index(duration)))
        triginfo.append(
            ('template', output_index, self.used_templates[output_index],
             self._time_to_index(duration), 0))
        if use_scale:
            triginfo.append(('use_scale', output_index, self._time_to_index(duration)))

        # load templates and store template trigger information
        index = self.gqs.output_index(output_index,
                                      int(self.used_templates[output_index]))
        self.gqs.setup_template(index, np.concatenate((np.zeros(8), template)))
        self.modulate += 2**(output_index * 16 +
                             self.used_templates[output_index])
        self.used_templates[output_index] += 1

        return triginfo

    def _update_duration(self, triginfo, duration):
        if len(triginfo) == 2:
            use_scale = False
        elif len(triginfo) == 3:
            use_scale = True
        else:
            raise NotImplementedError("something wrong...")

        carrier_list = list(triginfo[0])
        carrier_list[2] = self._time_to_index(duration)
        triginfo[0] = tuple(carrier_list)

        template_list = list(triginfo[1])
        template_list[3] = self._time_to_index(duration)
        triginfo[1] = tuple(template_list)

        if use_scale:
            scale_list = list(triginfo[2])
            scale_list[2] = self._time_to_index(duration)
            triginfo[2] = tuple(scale_list)

    def setup_template(self, output_port, template, envelope=False, use_scale=False):
        """
        Setup waveform generator to output a template.
        """

        output_index = output_port - 1

        # make sure template is a np array
        template = np.atleast_1d(template).astype(np.float64)
        if template.max() > 1.:
            print(
                "*** WARNING: exceeding DA limits, values clipped to maximum.")
            template[template > 1.] = 1.
        if template.min() < -1.:
            print(
                "*** WARNING: exceeding DA limits, values clipped to minimum.")
            template[template < -1.] = -1.

        # scale from 1.0 full scale to adc units full scale
        template = np.int16(32767 * template)
        # print("template_int:", np.max(template))

        # Template must be a multiple of 8 samples
        if template.shape[0] % 8 > 0:
            template = np.concatenate(
                (template, np.zeros(8 - template.shape[0] % 8)))

        triginfo = []

        # if envelope:
        #     triginfo.append(('carrier', output_index, template.shape[0] // 8))
        if use_scale:
            triginfo.append(('use_scale', output_index, template.shape[0] // 8))

        # load templates and store template trigger information
        offset = 0
        while template.shape[0] > 0:
            if self.used_templates[output_index] > 15:
                raise RuntimeError(
                    'Not enough templates on output {}'.format(output_index))
            subtemplate_length = min(4096 - 8, template.shape[0])
            triginfo.append(
                ('template', output_index, self.used_templates[output_index],
                 subtemplate_length // 8, offset))
            offset += subtemplate_length // 8
            # print(type(offset))
            index = self.gqs.output_index(
                output_index, int(self.used_templates[output_index]))
            self.gqs.setup_template(
                index,
                np.concatenate((np.zeros(8), template[:subtemplate_length])))
            template = template[subtemplate_length:]
            if envelope:
                self.modulate += 2**(output_index * 16 +
                                     self.used_templates[output_index])
            self.used_templates[output_index] += 1

        return triginfo

    def perform_measurement(self,
                            period,
                            repeat_count,
                            num_averages,
                            print_time=False,
                            verbose=False,
                            sleep_func=None,
    ):
        """
        Setup timing sequence
        """
        if sleep_func is None:
            sleep_func = time.sleep
        sorted_by_all = sorted(
            self.seq,
            key=lambda x: (x[0], x[1][0], x[1][1] if 1 < len(x[1]) else None))
        self.seq = sorted_by_all
        if verbose:
            print("sequence:", self.seq)
        raw_sequence = []
        last = None
        for entry in self.seq:
            if entry == last:
                continue
            last = entry
            if verbose:
                print("\tentry:", entry)
            at_time, command = entry
            action = command[0]
            if action == 'store':
                raw_sequence.append([
                    self._time_to_index(at_time), command[1],
                    self.gqs.store_index(command[2])
                ])
            elif action == 'readout':
                raw_sequence.append([
                    self._time_to_index(at_time), 1,
                    self.gqs.readout_index()
                ])
            elif action == 'carrier':
                raw_sequence.append([
                    self._time_to_index(at_time),
                    int(command[2]),
                    self.gqs.wavegen_index(command[1])
                ])
            elif action == 'use_scale':
                raw_sequence.append([
                    # compensate so that it happens together with the template
                    self._time_to_index(at_time) + 28,
                    int(command[2]),
                    self.gqs.use_scale_index(command[1])
                ])
            elif action == 'template':
                raw_sequence.append([
                    self._time_to_index(at_time) + command[4],
                    int(command[3]),
                    self.gqs.output_index(command[1], command[2])
                ])
                at_time += int(command[2])

            elif action == 'next_freq':
                raw_sequence.append([
                    self._time_to_index(at_time), 1,
                    self.gqs.freq_index(command[1])
                ])

            elif action == 'next_scale':
                raw_sequence.append([
                    self._time_to_index(at_time), 1,
                    self.gqs.scale_index(command[1])
                ])

        if verbose:
            print(raw_sequence)
        tot_repetitions = repeat_count * num_averages
        nr_triggers = self.gqs.setup_measurement(self._time_to_index(period),
                                                 tot_repetitions, raw_sequence)
        if verbose:
            print("Number of triggers: {:d}".format(nr_triggers))
        if nr_triggers > MAX_TRIGGERS:
            raise ValueError(
                'Exceeded maximum number of triggers! Trying to use {:d}, while max is {:d}.'
                .format(nr_triggers, MAX_TRIGGERS))

        # modulate templates
        self.gqs.printreg(19, self.gqs._loreg,
                          (int(self.modulate >> 0)) & (2**64 - 1))
        self.gqs.printreg(19, self.gqs._hireg,
                          (int(self.modulate >> 64)) & (2**64 - 1))

        tot_samples = np.sum([num for (idx, num) in self.saves])
        dma_length = tot_samples * repeat_count * 4  # when averaging
        if verbose:
            print("dma_length:", dma_length)

        expected_runtime = period * tot_repetitions  # s
        if print_time:
            print("Expected runtime: {:s}".format(format_sec(expected_runtime)))

        if self.dry_run == False:
            self.rflockin.stop_average_quantum()
            self.rflockin.output_control(0xffffffff)
            self.rflockin.write_register(self.gqs._quantum_list_of_commands)
            self.rflockin.output_control(0)
            self.rflockin.start_average_quantum(dma_length)
            t_start = time.time()
            t_end_exp = t_start + expected_runtime

            nold = self.rflockin.read_register(3 * 2)
            count = 0
            prev_print_len = 0
            while True:
                count += 1
                if print_time and (count == 10):
                    count = 0
                    msg = "Time left: {:s}".format(format_sec(t_end_exp - time.time()))
                    print_len = len(msg)
                    if print_len < prev_print_len:
                        msg += " " * (prev_print_len - print_len)
                    msg += "\r"
                    print(msg, end="")
                    prev_print_len = print_len
                # time.sleep(np.pi / 3 / 10)
                sleep_func(np.pi / 3 / 10)
                nnew = self.rflockin.read_register(3 * 2)
                if nnew == nold:
                    break
                nold = nnew
            if print_time:
                msg = "Total time: {:s}".format(format_sec(time.time() - t_start))
                print_len = len(msg)
                if print_len < prev_print_len:
                    msg += " " * (prev_print_len - print_len)
                print(msg)

            data = self.rflockin.read_buffer(dma_length)
            self.rflockin.stop_average_quantum()

            # scale to +-1 of input range
            data = data / 32767 / num_averages

            # reshape
            num_stores = repeat_count * len(self.saves) // len(
                self.store_ports)
            num_ports = len(self.store_ports)
            smpls_per_store = 8 * self._time_to_index(self.store_duration)
            data.shape = (num_stores, num_ports, smpls_per_store)

            # create time array
            t_arr = np.arange(smpls_per_store) / 4e9

            # return only the data the user wanted
            user_smpls_per_store = int(round(self.user_store_duration * self.sampling_freq))
            t_arr = t_arr[:user_smpls_per_store]
            data = data[:, :, :user_smpls_per_store]

            return t_arr, data

    def output_pulse(self, at_time, templates):
        for t in templates:
            for i in t:
                self.seq.append([at_time, i])

    def output_carrier(self, at_time, duration, port_list):
        duration_index = self._time_to_index(duration)
        for port in port_list:
            output_index = port - 1
            self.seq.append([at_time, ('carrier', output_index, duration_index)])

    def store(self, at_time):
        if self.store_duration is None:
            raise RuntimeError('did you forget to run set_store_duration?')
        if self.store_ports is None:
            raise RuntimeError('did you forget to run set_store_ports?')

        duration_cycles = self._time_to_index(self.store_duration)
        for port in self.store_ports:
            index = port - 1
            self.seq.append([at_time, ('store', duration_cycles, index)])
            self.saves.append((index, duration_cycles * 8))

        self._readout(at_time + 8 * 2e-9)  # delay transfer by 8 clk cycles

    def set_store_duration(self, T):
        T = float(T)
        if T > 4096e-9 or T < 0.0:
            raise ValueError("T must be in [0.0, 4096.0] ns")
        self.user_store_duration = T
        # round up to multiple of 512 ns, i.e. 256 clock cycles
        n = int(np.ceil(self.user_store_duration / 512e-9))
        self.store_duration = n * 512e-9

    def set_store_ports(self, store_ports):
        store_ports = np.atleast_1d(store_ports).astype(np.int64)
        if store_ports.min() < 1 or store_ports.max() > 8:
            raise ValueError('valid input ports are 1-8')
        self.store_ports = store_ports

    def _readout(self, at_time):
        self.seq.append([at_time, ('readout', )])

    def next_frequency(self, at_time, out_port_list):
        for out_port in out_port_list:
            self.seq.append([at_time, ('next_freq', out_port - 1)])

    def next_scale(self, at_time, out_port_list):
        # the change of scale happens 25 clock cycles before the change of a
        # template, so we delay the scale by 25 clock cycles here to compensate
        T = at_time + 25 * 2e-9
        for out_port in out_port_list:
            self.seq.append([T, ('next_scale', out_port - 1)])
