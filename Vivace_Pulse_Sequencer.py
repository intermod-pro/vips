# Authored by Johan Blomberg and Gustav Grännsjö, 2020

import os
from pathlib import Path
import re
import math
import time

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

from BaseDriver import LabberDriver, Error

import vivace
from vivace import pulsed
import envelopes

class Driver(LabberDriver):
    """ This class implements a Labber driver"""

    # In order to use the driver with actual hardware, DRY_RUN needs to be False
    DRY_RUN = True

    # Version numbers
    VIPS_VER = None
    VIVACE_FW_VER = None
    VIVACE_SERVER_VER = None
    VIVACE_API_VER = None

    """
    Debug mode is enabled with DEBUG_ENABLE = True. In Debug mode, every call to a SimpleQ method
    will be written to a log file. This lets the developer see if the given set of instrument
    settings correctly translate to the desired functionality.
    The file is written to C:/Users/[username]/DEBUG_PATH/DEBUG_FILE_NAME. If no
    file name is given, it will default to 'log.txt'.
    """
    DEBUG_ENABLE = False
    USER_DIR = os.path.expanduser('~')
    DEBUG_PATH = 'Vivace_Sequencer_Debug'
    DEBUG_FILE_NAME = ''

    def __init__(self, dInstrCfg=None, dComCfg=None, dValues=None, dOption=None,
                 dPrefs={}, queueIn=None, queueOut=None, logger=None):
        LabberDriver.__init__(self, dInstrCfg, dComCfg, dValues, dOption,
                              dPrefs, queueIn, queueOut, logger)

        # IP address used to connect to board
        self.address = self.getAddress()

        # Get the relevant version numbers
        self.fetch_version_numbers()

        self.sample_freq = None
        self.averages = None
        self.measurement_period = None
        self.iterations = None
        self.templates = [[[None for _ in range(16)] for _ in range(2)] for _ in range(8)]
        self.drag_templates = []
        self.drag_parameters = []
        self.template_defs = None
        self.port_settings = None
        self.pulse_definitions = None
        self.pulse_id_counter = 0
        self.amp_matrix = None
        self.fp_matrix = None
        self.carrier_changes = None
        self.samples_per_iteration = None

        # Sample ports
        self.store_ports = None

        # Measurement output
        self.time_array = None
        self.results = None

        # This list is used to keep track of the specific options used when getting traces in Labber
        self.previously_outputted_trace_configs = []

        self.debug_contents = ''
        self.INITIAL_TIME = None

    def reset_instrument(self):
        """
        Reinitialise the driver's state. Should be equivalent to restarting the instrument.
        """
        self.averages = None
        self.measurement_period = None
        self.iterations = None
        self.templates = [[[None for _ in range(16)] for _ in range(2)] for _ in range(8)]
        self.drag_templates = []
        self.drag_parameters = []
        self.template_defs = None
        self.port_settings = None
        self.pulse_definitions = None
        self.pulse_id_counter = 0
        self.amp_matrix = None
        self.fp_matrix = None
        self.carrier_changes = None
        self.samples_per_iteration = None

        # Sample ports
        self.store_ports = None

        # Measurement output
        self.time_array = None
        self.results = None

        self.previously_outputted_trace_configs = []

        self.debug_contents = ''
        self.INITIAL_TIME = None

    def get_next_pulse_id(self):
        self.pulse_id_counter += 1
        return self.pulse_id_counter - 1

    def performOpen(self, options={}):
        """Perform the operation of opening the instrument connection"""

    def performClose(self, bError=False, options={}):
        """Perform the close instrument connection operation"""

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        """
        Perform the Set Value instrument operation.
        This method will reinitialise the instrument if any quant that affects the board is changed.
        It is also responsible for type conversion and input formatting.
        """
        set_commands = quant.set_cmd.replace(' ', '').split(',')

        # If a new value is set that affects the outcome of a measurement, we need to
        # reset our stored variables to force a re-run of our measurements
        if 'not_affecting_board' not in set_commands:
            if self.is_value_new(quant, value):
                self.reset_instrument()

        # Some quants should only accept input of the form [base] + [delta]*i
        if 'time_string' in set_commands:
            # Strip the input down to the essential part
            value = value.replace('INVALID: ', '')
            input_str = value.replace(' ', '')
            # The representation of a number in E notation
            num_rex = r'(([0-9]+)|([0-9]*\.[0-9]+))(e-?[0-9]+)?'
            rex = re.compile(num_rex + '(\\*?i|\\+' + num_rex + '\\*?i)?', re.I)
            # Split the string if the quant accepts multiple values
            if 'single' in set_commands:
                strings = [input_str]
            else:
                strings = input_str.split(',')

            # Every string has to be valid
            for idx, s in enumerate(strings):
                match = rex.fullmatch(s)
                if not match:
                    return 'INVALID: ' + value
                # Split into the separate numbers to do some formatting
                num_strings = s.split('+')
                for idx2, num_string in enumerate(num_strings):
                    # If we have leading zeroes, truncate
                    while num_string.startswith('0') and len(num_string) > 1 and num_string[1].isnumeric():
                        num_string = num_string[1:]
                    # Insert a zero if the number starts with a period
                    if num_string.startswith('.'):
                        num_string = '0' + num_string
                    num_strings[idx2] = num_string
                strings[idx] = '+'.join(num_strings)
            return ', '.join(strings).replace('e', 'E')

        # Some quants should allow a list of comma-separated doubles
        if 'double_list' in set_commands:
            value = value.replace('INVALID: ', '')
            return self.parse_list_of_doubles(value)

        # Some Double quants should only allow integer values
        if 'int' in set_commands:
            return int(value)

        # The quant used for setting padding length allows values in intervals of 0.25
        if 'quarter_value' in set_commands:
            return round(value * 4) / 4

        # Version numbers should be kept constant
        if 'vips_version' in set_commands:
            return self.VIPS_VER

        if 'vivace_fw_version' in set_commands:
            return self.VIVACE_FW_VER

        if 'vivace_server_version' in set_commands:
            return self.VIVACE_SERVER_VER

        if 'vivace_api_version' in set_commands:
            return self.VIVACE_API_VER

        return value

    def is_value_new(self, quant, value):
        """
        Check if the given value differs from the value stored in the given quant.
        """
        current_value = self.getValue(quant.name)

        # Combo quants have datatype 2
        if quant.datatype == 2 and isinstance(value, float):
            current_value = self.getValueIndex(quant.name)

        # If it is a vector, we need to do a different equality test (because numpy does not work with == checks)
        if isinstance(value, dict):
            if len(current_value['y']) != len(value['y']):
                return True
            # If the vectors are of the same length and use plain time axes, we need to run an elementwise comparison
            if 'x' in current_value and 'x' in value:
                if not (np.allclose(current_value['y'], value['y'])
                        and np.allclose(current_value['x'], value['x'])):
                    return True
            # If the new vector is not in the same format, it counts as changed
            elif 'x' in current_value or 'x' in value:
                return True
            # The vectors are in base-delta time format, compare these values along with y values
            else:
                if not (np.allclose(current_value['y'], value['y'])
                        and np.isclose(current_value['t0'], value['t0'])
                        and np.isclose(current_value['dt'], value['dt'])):
                    return True
        # Use a little leniency when checking floats due to rounding errors in python
        elif isinstance(value, float):
            if not math.isclose(current_value, value):
                return True
        elif current_value is not value:
            return True
        return False

    def parse_number(self, string):
        """
        Parse time values on [base] + [delta]*i form and converts them to floats.
        Return both base and delta as separate floats.
        """
        string = string.lower().replace(' ', '')
        # String was in incorrect format
        if string.startswith('invalid'):
            raise ValueError('Invalid format of start time/duration string!')

        if '+' in string:
            # Both base and delta
            terms = string.split('+')
            base = float(terms[0])
            # Remove the '*i'
            tail_index = re.search("[*i]", terms[1]).start()
            delta = float(terms[1][:tail_index])
            return base, delta

        if 'i' in string:
            # only delta, remove the 'i' and the '*' if available
            tail_index = re.search("[*i]", string).start()
            return 0.0, float(string[:tail_index])

        # No + or i, we only have a base time
        return float(string), 0.0

    def parse_list_of_doubles(self, string):
        """
        Ensure that the given input string is a list of comma-separated floats.
        Return a formatted version of the input string, preceded by INVALID if something is incorrect.
        """
        for val in string.split(','):
            try:
                float(val)
            except ValueError:
                return 'INVALID: ' + string
        return string.replace(' ', '').replace(',', ', ').upper()

    def fetch_version_numbers(self):
        """
        Fetch the version number of both ViPS (from its definition file)
        and Vivace (by connecting to the hardware), and store them.
        """
        # Get ViPS version no.
        ini_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(ini_path, 'Vivace_Pulse_Sequencer.ini')) as ini:
            line = next(ini)
            while not line.startswith('version:'):
                line = next(ini)
            version = line.split(': ')[1]
            self.VIPS_VER = version

        # Get Vivace's version numbers
        with pulsed.Pulsed(dry_run=self.DRY_RUN, address=self.address) as q:
            self.VIVACE_API_VER = '1.0.2'  # TODO fetch from Vivace

            try:
                fw_ver = q._rflockin.read_register(24)
                server_ver = q._rflockin.get_server_version()
            except AttributeError:
                fw_ver = 'Could not connect to Vivace :('
                server_ver = 'Could not connect to Vivace :('
            self.VIVACE_FW_VER = fw_ver
            self.VIVACE_SERVER_VER = server_ver

    def performGetValue(self, quant, options={}):
        """
        Perform the Get Value instrument operation.
        Responsible for starting measurements and fetching and formatting the resulting time traces.
        Return the requested time trace in the form of a Labber Trace Dict.
        """

        if quant.get_cmd == 'get_result':

            # If we don't have any results, get some
            if self.results is None:
                self.perform_measurement()

            # See which trace we are getting
            iteration_idx = int(self.getValue('Index of displayed time trace - iteration')) - 1
            pulse_idx = int(self.getValue('Index of displayed time trace - sample pulse')) - 1
            window_idx = iteration_idx * self.samples_per_iteration + pulse_idx

            # We construct a unique "fingerprint" of the circumstance in which we called performGetValue.
            # If we get a repeated circumstance, we can assume that the user would want a new measurement.
            # The 'delay' value is almost always different, so we force it to 0
            options['delay'] = 0
            circumstance = (quant.name, window_idx, options)
            # Add the circumstance information to the debug log
            self.add_debug_line(str(circumstance))
            if circumstance in self.previously_outputted_trace_configs:
                self.reset_instrument()
                self.perform_measurement()
                # Also reset the list, only keeping the current config
                self.previously_outputted_trace_configs = [circumstance]
            else:
                self.previously_outputted_trace_configs.append(circumstance)

            if not self.DRY_RUN:
                return self.get_trace(quant)
            else:
                # Dummy output
                n_points = 500
                high_low = np.linspace(1, -1, n_points)
                low_high = np.linspace(-1, 1, n_points)
                dummy_y = np.empty(n_points * 2)
                dummy_y[0::2] = high_low
                dummy_y[1::2] = low_high
                times = np.linspace(-1, 1, n_points * 2)
                return quant.getTraceDict(dummy_y, x=times, t0=times[0], dt=(times[1] - times[0]))
        if quant.get_cmd == 'template_preview':
            return self.get_template_preview(quant)
        if quant.name == 'Pulse sequence preview':
            return self.get_sequence_preview(quant)

        return quant.getValue()

    def get_template_preview(self, quant):
        """
        Construct a wave based on the envelope template
        indicated by the given quant, and return it.
        """
        with pulsed.Pulsed(ext_ref_clk=True, dry_run=True, address=self.address) as q:
            template_defs = self.get_template_defs(q)
            # The template number X is the first character in the second word in "Template X: Preview"
            template_no = int(quant.name.split()[1][0])
            template_def = template_defs[template_no - 1]
            x, y = self.template_def_to_points(template_def, 0, q)

            # Add a fake 0 at the end so the template looks nicer
            x = np.append(x, x[-1] + 1 / q.sampling_freq)
            y = np.append(y, 0)

            return quant.getTraceDict(y, x=x, t0=x[0], dt=(x[1] - x[0]))

    def template_def_to_points(self, template_def, iteration, q):
        """
        Calculate X and Y point values for the given template definition and iteration.
        Return X, Y.
        """
        if 'Rise Points' in template_def:  # Gauss long drive
            duration = template_def['Base'] + template_def['Delta'] * iteration
            rise_p = template_def['Rise Points']
            n_long_points = int(round((duration - 2 * template_def['Flank Duration']) * q.sampling_freq))
            if n_long_points >= 1e5:
                raise ValueError('Your long drive template is too long to preview, '
                                 'it would take forever to create a list of >100 000 points!')
            long_p = np.linspace(1, 1, n_long_points)
            fall_p = template_def['Fall Points']
            total_points = len(rise_p) + n_long_points + len(fall_p)
            end_point = (total_points - 1) / q.sampling_freq
            x = np.linspace(0, end_point, total_points)
            y = np.concatenate((rise_p, long_p, fall_p))
            return x, y

        if 'Base' in template_def:  # Long drive
            duration = template_def['Base'] + template_def['Delta'] * iteration
            if duration <= 0:
                return [], []
            n_points = int(round(duration * q.sampling_freq))
            if n_points >= 1e5:
                raise ValueError('Your long drive template is too long to preview, '
                                 'it would take forever to create a list of >100 000 points!')
            end_point = (n_points - 1) / q.sampling_freq
            x = np.linspace(0, end_point, n_points)
            y = np.linspace(1, 1, n_points)
            return x, y

        if template_def['Duration'] <= 0:
            return [], []
        y = template_def['Points']
        end_point = (len(y) - 1) / q.sampling_freq
        x = np.linspace(0, end_point, len(y))
        return x, y

    def get_sequence_preview(self, quant):
        """
        Construct a waveform from the pulse sequence information in the instrument
        and return a TraceDict with its information.
        """
        with pulsed.Pulsed(ext_ref_clk=True, dry_run=True, address=self.address) as q:
            self.setup_instrument(q)

            period = self.getValue('Trigger period')
            preview_port = int(self.getValue('Preview port'))
            preview_iter = int(self.getValue('Preview iteration') - 1)
            preview_samples = self.getValue('Preview sample windows')
            use_slice = self.getValue('Enable preview slicing')
            slice_start = self.getValue('Preview slice start')
            slice_end = min(self.getValue('Preview slice end'), period)
            # Display nothing if the requested index is too high
            if preview_iter >= self.iterations:
                return None

            # The preview will take the form of a list of points, with length determined by trigger period
            sampling_freq = int(q.sampling_freq)
            preview_points = np.zeros(int(sampling_freq * period) + 1)
            sample_pulses = []
            for pulse in self.pulse_definitions:
                # Sample pulses will be handled separately
                if 'Sample' in pulse:
                    if preview_samples and preview_port in self.store_ports:
                        sample_pulses.append(pulse)
                    else:
                        continue
                else:
                    pulse_port = pulse['Port']
                    if pulse_port != preview_port:
                        continue

                    # Get pulse's envelope
                    template_no = pulse['Template_no']
                    template_def = self.template_defs[template_no - 1]

                    # If we have DRAG pulses on this port, get the modified envelopes
                    if 'DRAG_idx' in pulse:
                        templ_x, _ = self.template_def_to_points(template_def, preview_iter, q)
                        templ_y = self.drag_templates[pulse['DRAG_idx']][1]
                    else:
                        templ_x, templ_y = self.template_def_to_points(template_def, preview_iter, q)
                    if len(templ_y) == 0:
                        continue

                    # Get other relevant parameters
                    start_base, start_delta = pulse['Time']
                    time = start_base + start_delta * preview_iter
                    abs_time = self.get_absolute_time(start_base, start_delta, preview_iter)
                    p_amp, p_freq, p_phase = self.get_amp_freq_phase(pulse, preview_iter)

                    # Calculate phase relative to latest carrier reset.
                    reset_time = -1
                    for (t, _, _) in self.carrier_changes[pulse_port-1]:
                        if t > abs_time:
                            break
                        reset_time = t
                    p_phase = self.phase_sync(p_freq, p_phase, abs_time - reset_time)
                    # Construct the pulse
                    if p_freq != 0 and pulse['Carrier'] != 0:
                        carrier = np.sin(2*np.pi * p_freq * templ_x + np.pi*p_phase)
                        templ_y = templ_y * carrier
                    wave = templ_y * p_amp

                    # Place it in the preview timeline
                    pulse_index = int(time * sampling_freq)
                    points_that_fit = len(preview_points[pulse_index:(pulse_index+len(wave))])
                    preview_points[pulse_index:(pulse_index + points_that_fit)] += wave[:points_that_fit]

            # Display the sample windows
            for sample in sample_pulses:
                start_base, start_delta = sample['Time']
                start = start_base + start_delta * preview_iter
                duration = self.getValue('Sampling - duration')
                wave = np.linspace(-0.1, -0.1, duration * sampling_freq)
                window_index = int(start * sampling_freq)
                points_that_fit = len(preview_points[window_index:(window_index+len(wave))])
                preview_points[window_index:(window_index + len(wave))] = wave[:points_that_fit]

        self.reset_instrument()
        if use_slice:
            start_idx = int(slice_start * sampling_freq)
            end_idx = int(slice_end * sampling_freq)
            if end_idx - start_idx <= 0:
                return None

            preview_points = preview_points[start_idx:end_idx+1]
            times = np.linspace(slice_start, slice_end, len(preview_points))
        else:
            times = np.linspace(0, period, len(preview_points))

        return quant.getTraceDict(preview_points, x=times, t0=times[0], dt=(times[1] - times[0]))

    def get_trace(self, quant):
        """
        Fetch the requested time trace(s) from the most recent result matrix.
        Return it in the form of a Labber Trace Dict
        """
        # Find the port number
        port_no = None
        for char in quant.name:
            if char.isdigit():
                port_no = int(char)
                break
        if port_no is None:
            raise ValueError('Could not find a port number to output for')
        # Convert port number to index of active sampling ports
        try:
            output_idx = self.store_ports.index(port_no)
        except ValueError:
            # The port in question wasn't active, so don't display anything
            return None

        # Get which sample window to show
        iteration_idx = int(self.getValue('Index of displayed time trace - iteration')) - 1
        pulse_idx = int(self.getValue('Index of displayed time trace - sample pulse')) - 1
        window_idx = iteration_idx * self.samples_per_iteration + pulse_idx
        if pulse_idx >= self.samples_per_iteration or iteration_idx >= self.iterations:
            # If the user requests a window which doesn't have a trace, don't show anything
            return None

        measurement = self.results[window_idx][output_idx]
        return quant.getTraceDict(measurement, x=self.time_array, t0=self.time_array[0],
                                  dt=(self.time_array[1] - self.time_array[0]))

    def perform_measurement(self):
        """
        Use all given information in the instrument to perform a measurement with the board.
        Store the resulting output in a global variable.
        """
        with pulsed.Pulsed(ext_ref_clk=True, dry_run=self.DRY_RUN, address=self.address) as q:
            self.setup_instrument(q)

            # Set up our actual LUTs on the board
            self.apply_LUTs(q)

            # Set up the full pulse on the board
            self.setup_sequence(q)

            # Start measuring
            total_time = self.measurement_period * (self.iterations + 1)
            self.add_debug_line('q.perform_measurement()')
            output = q.perform_measurement(total_time, 1, self.averages)
            if not q.dry_run:
                # Store the results
                (t_array, result) = output
                self.time_array = list(t_array)
                self.results = result
            else:
                self.results = 'Dummy result'
            if self.DEBUG_ENABLE:
                self.print_lines()

    def setup_instrument(self, q):
        self.sample_freq = q.sampling_freq
        # Get some general parameters such as no. of averages, trigger period etc.
        self.get_general_settings()
        # Get template definitions
        self.template_defs = self.get_template_defs(q)
        # Port settings
        self.port_settings = self.get_port_settings()
        # Set DC biases for all ports
        self.set_dc_biases(q)
        # Pulse definitions
        self.pulse_definitions = self.get_all_pulse_defs(q)
        # Sampling
        sample_definitions = self.get_sample_pulses(q)
        self.pulse_definitions.extend(sample_definitions)
        # Copy pulse definitions on specified ports
        self.copy_defs(q)
        # Sort our definitions chronologically
        self.pulse_definitions = sorted(self.pulse_definitions,
                                        key=lambda x: x['Time'][0] + x['Time'][1])
        self.validate_pulse_definitions()
        # Get the values that will go in the LUTs
        self.amp_matrix, self.fp_matrix, self.carrier_changes = self.get_LUT_values()

    def get_general_settings(self):
        """
        Get instrument parameters from the 'General settings' section.
        These are saved in global variables.
        """
        self.averages = int(self.getValue('Average'))
        self.measurement_period = self.getValue('Trigger period')
        if self.measurement_period == 0:
            raise ValueError('Trigger period cannot be 0!')
        self.iterations = int(self.getValue('Iterations'))

    def get_template_defs(self, q):
        """
        Get the user-defined templates (consisting of shapes and durations).
        These are represented as dictionaries containing a 'Points' value representing the
        template's shape and a 'Duration' value.
        Long drive templates are a special case, and only contain duration values in the form of a
        'Base' and a 'Delta' value by default. They will also contain 'Rise Points' and 'Fall Points' if
        Gaussian flanks are enabled.
        Return the template definitions in the form of a list.
        """
        num_templates = self.getValue('Envelope template count')
        template_defs = [{} for _ in range(15)]
        for def_idx in range(1, int(num_templates) + 1):
            template_name = self.getValue(f'Envelope template {def_idx}: shape')

            # Long drive templates are a special case
            if template_name == 'Long drive':
                template = self.get_long_drive_definition(def_idx, q.sampling_freq)
            else:
                template = {}
                # Other types share a lot of behaviour
                duration = self.getValue(f'Envelope template {def_idx}: duration')
                template['Duration'] = duration
                n_points = int(round(duration * q.sampling_freq))
                use_padding = self.getValue(f'Envelope template {def_idx}: use zero-padding')
                template['Points'] = self.get_template_points(template_name, n_points, def_idx)

                # Pad with leading zeroes if requested
                if use_padding:
                    pad_length = self.getValue(f'Envelope template {def_idx}: padding length')
                    pad_points = int(pad_length * 4)
                    template['Points'] = np.concatenate((np.zeros(pad_points), template['Points']))

            template_defs[def_idx - 1] = template
        return template_defs

    def get_long_drive_definition(self, definition_idx, sampling_frequency):
        """
        Construct and return a template definition for a long drive, based on the user-set parameters on
        definition number definition_idx in the instrument.
        """
        template = {}
        dur_string = self.getValue(f'Envelope template {definition_idx}: long drive duration')
        try:
            template['Base'], template['Delta'] = self.parse_number(dur_string)
        except ValueError:
            error_msg = f'Invalid duration value for template definition {definition_idx}'
            raise ValueError(error_msg)
        # Check if we should add gaussian flanks
        use_gaussian = self.getValue(f'Envelope template {definition_idx}: use gaussian rise and fall')
        if use_gaussian:
            flank_duration = self.getValue(f'Envelope template {definition_idx}: gaussian rise and fall duration')
            if flank_duration * 2 > template['Base']:
                raise ValueError(f'The rise and fall durations in template {definition_idx} exceed the '
                                 f'template\'s total duration!')
            template['Flank Duration'] = flank_duration
            flank_points = int(round(flank_duration * sampling_frequency))
            # How many sigma we should cut off our gaussian at
            cutoff = 3.2
            # Rise
            rise_x = np.linspace(-cutoff, 0, flank_points+1)
            rise_y = norm.pdf(rise_x, 0, 1)
            rise_y = rise_y / rise_y.max()
            rise_y[0] = 0  # For symmetry's sake
            template['Rise Points'] = rise_y[:-1]
            # Fall
            fall_x = np.linspace(0, cutoff, flank_points+1)
            fall_y = norm.pdf(fall_x, 0, 1)
            fall_y = fall_y / fall_y.max()
            template['Fall Points'] = fall_y[:-1]
        return template

    def get_template_points(self, template_name, n_points, definition_idx):
        """
        Return an n_points long list of points forming the shape corresponding to the given template_name.
        definition_idx is needed to fetch extra user-set parameters for certain templates (like the p in sinP).
        """
        if template_name == 'Square':
            return np.ones(n_points+1)[:-1]
        if template_name == 'SinP':
            p = self.getValue(f'Envelope template {definition_idx}: sinP Value')
            return envelopes.sin_p(p, n_points+1)[:-1]
        if template_name == 'Sin2':
            return envelopes.sin2(n_points+1)[:-1]
        if template_name == 'Sinc':
            cutoff = self.getValue(f'Envelope template {definition_idx}: sinc cutoff')
            return envelopes.sinc(cutoff, n_points+1)[:-1]
        if template_name == 'Triangle':
            return envelopes.triangle(n_points+1)[:-1]
        if template_name == 'Gaussian':
            trunc = self.getValue(f'Envelope template {definition_idx}: gaussian truncation')
            return envelopes.gaussian(n_points+1, trunc)[:-1]
        if template_name == 'Cool':
            return envelopes.cool(n_points+1)[:-1]
        if template_name.startswith('Custom'):
            idx = template_name[-1]
            # Fetch the template's shape from the designated input
            custom_template = self.getValue(f'Custom template {idx}')
            custom_values = custom_template['y']
            if len(custom_values) == 0:
                raise ValueError(f'Input for custom template {idx} does not contain any data!')
            if 'x' in custom_template:
                custom_times = custom_template['x']
            else:
                custom_times = np.linspace(custom_template['t0'],
                                           len(custom_values)*custom_template['dt']+custom_template['t0'],
                                           len(custom_values))

            # Rescale template to range [-1, +1]
            custom_values = custom_values / max(abs(custom_values))

            # Fit a curve to the fetched shape, and then set up the template based on this fitted curve
            curve_fit = interp1d(custom_times, custom_values)
            return curve_fit(np.linspace(custom_times[0], custom_times[-1], n_points))

        raise ValueError('Selected envelope shape is not defined in driver!')

    def get_port_settings(self):
        """
        Get each port's settings.
        Port settings are represented by an array with 8 dictionaries, each representing a single port's settings.
            The 'Mode' key contains the port's mode (Define, Disabled, Copy)
            If a port is in copy mode, also save the port it is copying from in a 'Sibling' key.
        Return a list of these dictionaries.
        """
        port_settings = [{} for _ in range(8)]

        # Get the mode for each port
        for port in range(1, 9):
            p = port - 1
            mode = self.getValue(f'Port {port} - mode')
            port_settings[p]['Mode'] = mode
            if mode == 'Copy':
                port_settings[p]['Sibling'] = int(self.getValue(f'Port {port} - copy sequence from'))

        return port_settings

    def set_dc_biases(self, q):
        if not q.dry_run:
            for port in range(1, 9):
                bias = self.getValue(f'Port {port} - DC bias')
                bias = bias / 1.25
                self.add_debug_line(f'q._rflockin.set_bias_dac(port={port}, bias={bias})')
                q._rflockin.set_bias_dac(port, bias)

    def get_all_pulse_defs(self, q):
        """
        Get the user-defined pulse sequence information for each port.
        Return a list of the pulse definition dictionaries.
        """
        pulse_definitions = []

        # Go through every port definition section one after another
        for port in range(1, 9):

            settings = self.port_settings[port - 1]

            # If no pulses are set up on the port, skip to the next
            if settings['Mode'] != 'Define':
                continue

            # Check how many pulses are defined
            n_pulses = int(self.getValue(f'Pulses for port {port}'))
            # Step through all pulse definitions
            for p_def_idx in range(1, n_pulses + 1):
                pulse_defs = self.create_pulse_defs(port, p_def_idx, q)
                pulse_definitions.extend(pulse_defs)
        return pulse_definitions

    def create_pulse_defs(self, port, def_idx, q):
        """
        Create and return a list of pulse definition dictionaries based on a single pulse definition in the instrument.
        If the user has entered multiple start times, one dictionary will be returned for every start time.

        Pulses are stored as dictionaries with the following entries:
            Time: A tuple of the format (base, delta) indicating the pulse's start time.
            Port: The port that the pulse is output on.
            Template_no: The number of the template used for the pulse.
            Amp: The pulse's amplitude scale.
            Freq: The pulse's carrier frequency.
            Phase: The pulse's phase.
        """
        template_no = int(self.getValue(f'Port {port} - def {def_idx} - template'))
        carrier = self.get_carrier_index(self.getValue(f'Port {port} - def {def_idx} - sine generator'))

        # Non-DRAG pulses can have their template set up normally
        if carrier != 3:
            try:
                self.setup_template(q, port, carrier, template_no)
            except ValueError as err:
                if str(err).startswith("No template def"):
                    err_msg = f'Pulse definition {def_idx} on port {port} uses an undefined template!'
                    raise ValueError(err_msg)
                raise err

        sweep_param = self.getValue(f'Port {port} - def {def_idx} - Sweep param')
        if sweep_param == 'Amplitude scale':
            amp = self.get_sweep_values(port, def_idx)
        else:
            amp = self.getValue(f'Port {port} - def {def_idx} - amp')
            amp = [amp] * self.iterations
        if sweep_param == 'Carrier frequency':
            freq = self.get_sweep_values(port, def_idx)
        else:
            freq = self.getValue(f'Port {port} - def {def_idx} - freq')
            freq = [freq] * self.iterations
        if sweep_param == 'Phase':
            phase = self.get_sweep_values(port, def_idx)
        else:
            phase = self.getValue(f'Port {port} - def {def_idx} - phase')
            phase = [phase] * self.iterations

        repeat_count = int(self.getValue(f'Port {port} - def {def_idx} - repeat count'))
        if repeat_count > 1:
            # Get the pulse's duration
            template_def = self.template_defs[template_no - 1]
            if 'Duration' not in template_def:
                raise ValueError(f'Pulse definition {def_idx} on port {port}: '
                                 f'Pulses that use Long drive envelopes cannot be repeated!')
            duration = template_def['Duration']
        else:
            duration = None

        start_times = self.getValue(f'Port {port} - def {def_idx} - start times')
        start_times = start_times.split(',')
        n_start_times = len(start_times)

        # We define a new pulse for every start time
        pulse_defs = []
        for i, start_time in enumerate(start_times):
            # Parse the original start times and make copies if requested
            if i < n_start_times:
                # Get start time value
                try:
                    time = self.parse_number(start_time)
                except ValueError:
                    err_msg = f'Invalid start time definition for port {port}, definition {def_idx}'
                    raise ValueError(err_msg)

                # Create repeats
                for r in range(1, repeat_count):
                    # Float addition can have some bad side effects, so we round to nearest ns
                    start_times.append((round(time[0] + duration * r, 9), time[1]))
            else:
                time = start_time

            # Pulses should only be output on even ns values
            base_nano = round(time[0] * 1e9, 4)
            delta_nano = round(time[1] * 1e9, 4)
            if base_nano % 2 != 0 or delta_nano % 2 != 0:
                raise ValueError(f'The starting time of pulse {def_idx} on port {port} is an odd number of nanoseconds '
                                 f'at some point. The board can currently only output pulses at '
                                 f'even nanosecond values.')

            # Save this pulse definition for later use
            if carrier != 3:
                pulse_defs.append({
                    'ID': self.get_next_pulse_id(),
                    'Time': time,
                    'Port': port,
                    'Carrier': carrier,
                    'Template_no': template_no,
                    'Amp': amp.copy(),
                    'Freq': freq.copy(),
                    'Phase': phase.copy()})

            # If the pulse is in DRAG mode, we need to calculate some extra parameters
            else:
                pulse_defs.extend(self.calculate_drag(
                    def_idx,
                    time,
                    port,
                    template_no,
                    amp.copy(),
                    freq.copy(),
                    phase.copy(),
                    q))

        return pulse_defs

    def get_carrier_index(self, option):
        if option == 'DRAG':
            return 3
        if option == 'None':
            return 0
        return int(option)

    def calculate_drag(self, def_idx, time, port, template_no, amp, freq, phase, q):
        """
        Creates four DRAG pulses based on a pulse definition set to DRAG mode.
        This will also result in the creation of four new templates on the board, which will be
        stored in self.drag_templates, at an index saved in each pulse definition's 'DRAG_idx' key.
        Returns a list of the four pulse definitions.
        """

        sibling_port = int(self.getValue(f'Port {port} - def {def_idx} - DRAG sibling port'))
        times, points = self.template_def_to_points(self.template_defs[template_no - 1], 0, q)
        scale = self.getValue(f'Port {port} - def {def_idx} - DRAG scale')
        detuning = self.getValue(f'Port {port} - def {def_idx} - DRAG detuning frequency')
        phase_shift = self.getValue(f'Port {port} - def {def_idx} - DRAG phase shift')

        if (template_no, scale, detuning) not in self.drag_parameters:

            beta = scale * q.sampling_freq

            # Add the original envelope's gradient (scaled) as a complex part
            complex_points = points + 1j * beta * np.gradient(points)
            complex_points = complex_points * np.exp(1j * 2 * np.pi * detuning * times)
            re_points = np.real(complex_points)
            im_points = np.imag(complex_points)

            # Rescale points to be within [-0.5, +0.5]
            biggest_outlier = 2 * max(max(abs(re_points)), max(abs(im_points)))
            re_points = re_points / biggest_outlier
            im_points = im_points / biggest_outlier

            # Set up both templates on the board and store them
            self.add_debug_line(f'q.setup_template(port={port}, points={re_points}, carrier=1, use_scale=True)')
            self.add_debug_line(f'q.setup_template(port={port}, points={im_points}, carrier=2, use_scale=True)')
            self.add_debug_line(f'q.setup_template(port={sibling_port}, points={re_points}, carrier=1, use_scale=True)')
            self.add_debug_line(f'q.setup_template(port={sibling_port}, points={im_points}, carrier=2, use_scale=True)')
            base_re_template = q.setup_template(port, re_points, 1, True)
            base_im_template = q.setup_template(port, im_points, 2, True)
            sibl_re_template = q.setup_template(sibling_port, re_points, 1, True)
            sibl_im_template = q.setup_template(sibling_port, im_points, 2, True)
            self.drag_templates.append((base_re_template, re_points))
            base_re_idx = len(self.drag_templates) - 1
            self.drag_templates.append((base_im_template, im_points))
            base_im_idx = len(self.drag_templates) - 1
            self.drag_templates.append((sibl_re_template, re_points))
            sibl_re_idx = len(self.drag_templates) - 1
            self.drag_templates.append((sibl_im_template, im_points))
            sibl_im_idx = len(self.drag_templates) - 1

        # We've already made the necessary templates, find their index
        else:
            match_idx = self.drag_parameters.index((template_no, scale, detuning))

            base_re_idx = match_idx * 4 + 0
            base_im_idx = match_idx * 4 + 1
            sibl_re_idx = match_idx * 4 + 2
            sibl_im_idx = match_idx * 4 + 3

        # Create four pulse defs
        pulse_defs = []
        for i in range(4):
            d_port = port if i in (0, 1) else sibling_port
            d_carrier = 1 if i in (0, 2) else 2

            # Phase offset and template index is different between every definition
            if i == 0:
                d_idx = base_re_idx
                # Cosine, so we need to shift our sine by pi/2
                d_phase = [p + 1/2 for p in phase.copy()]
            elif i == 1:
                d_idx = base_im_idx
                # Negative cosine with pi/2 offset, so 2pi = 0 in total
                d_phase = phase.copy()
            elif i == 2:
                d_idx = sibl_re_idx
                # Sine without offset
                d_phase = [p + phase_shift for p in phase.copy()]
            else:
                d_idx = sibl_im_idx
                # Negative sine with pi/2 offset, so 3/2pi in total
                d_phase = [p + phase_shift + 3/2 for p in phase.copy()]

            pulse_defs.append({
                'ID': self.get_next_pulse_id(),
                'Time': time,
                'Port': d_port,
                'Carrier': d_carrier,
                'Template_no': template_no,
                'DRAG_idx': d_idx,
                'Amp': amp.copy(),
                'Freq': freq.copy(),
                'Phase': d_phase}
            )

        return pulse_defs

    def setup_template(self, q, port, carrier, template_no):
        """
        Set up the specified template on the specified port on the board.
        Store the template globally in the format given by Vivace.
        """
        template_def = self.template_defs[template_no - 1]
        # Check that the given template number has a definition
        if len(template_def) == 0:
            raise ValueError("No template def found!")
        # If the requested template has not already been set up for this port, do it.
        if self.templates[port - 1][carrier - 1][template_no - 1] is None:
            try:
                # Only long drives have the 'Base' key
                if 'Base' in template_def:
                    initial_length = template_def['Base']
                    # Set up gaussian rise and fall templates if defined.
                    if 'Flank Duration' in template_def:
                        initial_length -= 2 * template_def['Flank Duration']
                        self.add_debug_line(f'q.setup_template(port={port}, points={template_def["Rise Points"]}, carrier={carrier}, use_scale=True)')
                        rise_template = q.setup_template(port, template_def['Rise Points'], carrier, use_scale=True)
                        self.add_debug_line(f'q.setup_template(port={port}, points={template_def["Fall Points"]}, carrier={carrier}, use_scale=True)')
                        fall_template = q.setup_template(port, template_def['Fall Points'], carrier, use_scale=True)
                    self.add_debug_line(f'q.setup_long_drive(port={port}, carrier={carrier}, duration={initial_length}, use_scale=True)')
                    try:
                        long_template = q.setup_long_drive(port,
                                                           carrier,
                                                           initial_length,
                                                           use_scale=True)
                    except ValueError as err:
                        if err.args[0].startswith('valid carriers'):
                            raise ValueError('Long drive envelopes have to be on either sine generator 1 or 2!')
                    if 'Flank Duration' in template_def:
                        self.templates[port - 1][carrier - 1][template_no - 1] = (rise_template, long_template, fall_template)
                    else:
                        self.templates[port - 1][carrier - 1][template_no - 1] = long_template
                else:
                    self.add_debug_line(f'q.setup_template(port={port}, points={template_def["Points"]}, carrier={carrier}, use_scale=True)')
                    self.templates[port - 1][carrier - 1][template_no - 1] = q.setup_template(port,
                                                                                 template_def['Points'],
                                                                                 carrier,
                                                                                 use_scale=True)
            except RuntimeError as error:
                if error.args[0].startswith('Not enough templates on output'):
                    raise RuntimeError(f'There are more than 16 templates in use on port {port}!\n '
                                       '(Templates longer than 1024 ns are split into multiple, '
                                       'unless they are of type "Long drive")')

    def get_sample_pulses(self, q):
        """
        Set up the user-defined sample pulses on the board.
        These are stored in a dictionary format, with the following entries:
            Time: The pulse's starting time, given as the tuple (base, delta).
            Sample: Always set to True. Only used as a flag to identify this pulse as a sample pulse.
        Return a list of these dictionaries.
        """
        sample_definitions = []

        # Check one port at a time
        sampling_ports = []
        for port in range(1, 9):
            use_port = self.getValue(f'Sampling on port {port}')
            if use_port:
                sampling_ports.append(port)
        if len(sampling_ports) == 0:
            raise ValueError('Sampling not set up on any port!')

        duration = self.getValue(f'Sampling - duration')
        # The board can only sample for 4096 ns at a time, so if the user wants longer, we need to split up the calls.
        if duration > 4096e-9:
            raise ValueError('Sampling duration must be in [0.0, 4096.0] ns')

        self.add_debug_line(f'q.set_store_duration({duration})')
        q.set_store_duration(duration)

        # Save the ports we want to sample on
        self.add_debug_line(f'q.set_store_ports({sampling_ports})')
        q.set_store_ports(sampling_ports)
        self.store_ports = sampling_ports

        # Get times and duration
        start_times_string = self.getValue(f'Sampling - start times')
        start_times = start_times_string.split(',')
        self.samples_per_iteration = len(start_times)

        # Store the sample pulse's defining parameters
        for start_time in start_times:
            try:
                time = self.parse_number(start_time)
            except ValueError:
                err_msg = f'Invalid start time definition for sampling!'
                raise ValueError(err_msg)
            # Get a unique pulse def id
            p_id = self.get_next_pulse_id()

            sample_definitions.append({
                'ID': p_id,
                'Time': time,
                'Sample': True})

        return sample_definitions

    def get_sweep_values(self, port, def_idx):
        """
        Calculate and return a list of parameter values to sweep over based on the given pulse's sweep settings.
        """
        sweep_format = self.getValue(f'Port {port} - def {def_idx} - Sweep format')
        # Custom is a special case, we just get the values directly
        if sweep_format == 'Custom':
            step_values = self.getValue(f'Port {port} - def {def_idx} - Sweep custom steps')
            string_list = step_values.split(',')
            if len(string_list) != self.iterations:
                raise ValueError(f'The number of custom values for pulse definition '
                                 f'{def_idx} on port {port} does not match the number of iterations!')
            values = []
            for string in string_list:
                values.append(float(string))
            return values
        # For linear, we need to calculate the full list of values
        if sweep_format == 'Linear: Start-End':
            interval_start = self.getValue(f'Port {port} - def {def_idx} - Sweep linear start')
            interval_end = self.getValue(f'Port {port} - def {def_idx} - Sweep linear end')
        else:  # Center-span
            center = self.getValue(f'Port {port} - def {def_idx} - Sweep linear center')
            span = self.getValue(f'Port {port} - def {def_idx} - Sweep linear span')
            interval_start = center - (span / 2)
            interval_end = center + (span / 2)

        return list(np.linspace(interval_start, interval_end, self.iterations))

    def validate_pulse_definitions(self):
        for p in range(8):
            prev_start = -1
            prev_duration = 0
            prev_carrier = -1
            prev_amp = 0
            for it in range(self.iterations):
                for pulse in [p for p in self.pulse_definitions if 'Sample' not in p]:
                    overlap_error = False
                    if pulse['Port'] != p+1:
                        continue
                    start = pulse['Time']
                    curr_start = self.get_absolute_time(start[0], start[1], it)
                    temp_def = self.template_defs[pulse['Template_no']-1]
                    curr_duration = self.get_tempdef_duration(temp_def, it)
                    curr_carrier = pulse['Carrier']
                    curr_amp = pulse['Amp'][it]
                    if curr_start == prev_start:
                        if 'DRAG_idx' not in pulse and abs(curr_amp + prev_amp) > 1:
                            raise ValueError(f'The combined amplitude of the overlapping pulses at time '
                                             f'{start[0] + start[1]*it} on port {p + 1} exceeds 1 or -1!')

                        if (curr_carrier == prev_carrier or
                                curr_duration != prev_duration):
                            overlap_error = True
                    elif curr_start - (prev_start+prev_duration) < -1e-9:
                        overlap_error = True
                    if overlap_error:
                        raise ValueError(f"Two pulses overlap incorrectly at time {start[0] + start[1]*it} "
                                         f"on port {p + 1} during iteration {it+1}! "
                                         f"Overlapping pulses must use different carrier generators and "
                                         f"have identical start times and durations.")
                    prev_carrier = curr_carrier
                    prev_start = curr_start
                    prev_duration = curr_duration
                    prev_amp = curr_amp

    def get_tempdef_duration(self, tempdef, iteration):
        if 'Base' in tempdef:  # Long drive
            return tempdef['Base'] + tempdef['Delta'] * iteration
        else:
            return tempdef['Duration']

    def get_LUT_values(self):
        """
        Constructs matrices of amplitude and frequency/phase values from the pulse definitions.
        The matrices are indexed by port number, and will be the basis for the LUTs on the board.
        If all pulses have fixed amp/freq/phase values, each value is only present once in the list.
        If a sweep pulse is defined, the list will be a complete chronological sequence of values,
        so that the board will only ever need to step once in the LUT between pulses.
        Return two matrices: one for the amplitude scale values and one for tuples of frequency and phase values.
        """
        # Sample pulses are not relevant here, so take them out of the definition list
        pulse_definitions_no_samples = [pulse for pulse in self.pulse_definitions if 'Sample' not in pulse]

        # Divide the pulses by the port they're on
        port_timelines = [[] for _ in range(8)]
        for pulse in pulse_definitions_no_samples:
            p_port_idx = pulse['Port'] - 1
            port_timelines[p_port_idx].append(pulse)

        amp_values = [[] for _ in range(8)]
        freq_phase_values = [[] for _ in range(8)]
        port_carrier_changes = [[] for _ in range(8)]

        for port_idx, timeline in enumerate(port_timelines):
            # Don't do anything for ports without definitions
            if len(timeline) < 1:
                continue

            carrier_changes = []

            latest_fp1 = None
            latest_fp2 = None
            latest_change = None
            reference_times = {}

            # Step through each iteration to put together the LUTs in the order the values will be needed.
            for i in range(self.iterations):

                # Add any left over values from previous iteration
                if i > 0:
                    if latest_fp2 is None:
                        latest_fp2 = (0, 0)
                    if latest_fp1 is None:
                        latest_fp1 = (0, 0)
                    carrier_changes.append((latest_change,
                                            (latest_fp1[0], latest_fp1[1]),
                                            (latest_fp2[0], latest_fp2[1])))

                # Reset for the new iteration
                latest_fp1 = None
                latest_fp2 = None
                # The first carrier change will happen at the first pulse
                first_pulse_start = timeline[0]['Time']
                latest_change = self.get_absolute_time(first_pulse_start[0], first_pulse_start[1], i)
                reference_times = {}

                for pulse in timeline:
                    # Get some necessary parameters from the pulse's definition.
                    p_amp, p_freq, p_phase = self.get_amp_freq_phase(pulse, i)
                    time = pulse['Time']
                    abs_time = self.get_absolute_time(time[0], time[1], i)

                    # Save the amplitude value if it is new
                    if p_amp not in amp_values[port_idx]:
                        amp_values[port_idx].append(p_amp)

                    # If the pulse isn't using a carrier wave, it won't use LUT values
                    carrier = pulse['Carrier']
                    if carrier == 0:
                        continue

                    if p_freq in reference_times:
                        ref_start = reference_times[p_freq]
                    else:
                        reference_times[p_freq] = abs_time
                        ref_start = abs_time

                    if carrier == 1:
                        # If we have a free slot for carrier 1, save it and move on
                        if latest_fp1 is None:
                            # Calculate the phase difference between this pulse and the start of the carrier change
                            carr_change_ps = ((abs_time - latest_change) * p_freq * 2)
                            # Phase sync to the reference point, then subtract the potential "double generator" offset
                            ph = self.phase_sync(p_freq, p_phase, abs_time - ref_start) - carr_change_ps
                            for pul in self.pulse_definitions:
                                if pul['ID'] == pulse['ID']:
                                    pul['Phase'][i] = ph
                                    break
                            latest_fp1 = (p_freq, ph)
                            continue
                        # If this pulse uses the same values as the last saved ones, we don't need a swap
                        elif latest_fp1 == (p_freq, p_phase):
                            continue
                        # New freq/phase, we need a reset
                        else:
                            # If we never found any pulses for carrier 2 before we needed a swap, use dummy values
                            if latest_fp2 is None:
                                latest_fp2 = (0, 0)
                            carrier_changes.append((latest_change,
                                                    (latest_fp1[0], latest_fp1[1]),
                                                    (latest_fp2[0], latest_fp2[1])))
                            # Save the time that we will need access to these new freq/phase values.
                            latest_change = self.get_absolute_time(time[0], time[1], i)

                            # Update global pulse definition with new phase synced value
                            ph = self.phase_sync(p_freq, p_phase, abs_time - ref_start)
                            for pul in self.pulse_definitions:
                                if pul['ID'] == pulse['ID']:
                                    pul['Phase'][i] = ph
                                    break
                            latest_fp1 = (p_freq, ph)

                            # Carrier 2 now has a free slot
                            latest_fp2 = None
                    elif carrier == 2:  # Same for carrier 2
                        if latest_fp2 is None:
                            # Calculate the phase difference between this pulse and the start of the carrier change
                            carr_change_ps = ((abs_time - latest_change) * p_freq * 2)
                            # Phase sync to the reference point, then subtract the potential "double generator" offset
                            ph = self.phase_sync(p_freq, p_phase, abs_time - ref_start) - carr_change_ps
                            for pul in self.pulse_definitions:
                                if pul['ID'] == pulse['ID']:
                                    pul['Phase'][i] = ph
                                    break
                            latest_fp2 = (p_freq, ph)
                            continue
                        elif latest_fp2 == (p_freq, p_phase):
                            continue
                        else:
                            if latest_fp1 is None:
                                latest_fp1 = (0, 0)
                            carrier_changes.append((latest_change,
                                                    (latest_fp1[0], latest_fp1[1]),
                                                    (latest_fp2[0], latest_fp2[1])))
                            latest_change = self.get_absolute_time(time[0], time[1], i)

                            ph = self.phase_sync(p_freq, p_phase, abs_time - ref_start)
                            for pul in self.pulse_definitions:
                                if pul['ID'] == pulse['ID']:
                                    pul['Phase'][i] = ph
                                    break
                            latest_fp2 = (p_freq, ph)

                            latest_fp1 = None

            # Lock in the final freq/phase values if there are any left
            if latest_fp1 is not None or latest_fp2 is not None:
                if latest_fp2 is None:
                    latest_fp2 = (0, 0)
                if latest_fp1 is None:
                    latest_fp1 = (0, 0)
                carrier_changes.append((latest_change,
                                        (latest_fp1[0], latest_fp1[1]),
                                        (latest_fp2[0], latest_fp2[1])))
            port_carrier_changes[port_idx] = carrier_changes
            # Extract unique fp pairs from the change list into a LUT
            for (_, fp1, fp2) in carrier_changes:
                # If this value has not already been recorded in the LUT, add it.
                if not any([self.are_fp_pairs_close((fp1, fp2), fpfp) for fpfp in freq_phase_values[port_idx]]):
                    freq_phase_values[port_idx].append((fp1, fp2))

        return amp_values, freq_phase_values, port_carrier_changes

    def are_fp_pairs_close(self, tuptup1, tuptup2):
        return (math.isclose(tuptup1[0][0], tuptup2[0][0]) and
                math.isclose(tuptup1[0][1], tuptup2[0][1]) and
                math.isclose(tuptup1[1][0], tuptup2[1][0]) and
                math.isclose(tuptup1[1][1], tuptup2[1][1]))

    def phase_sync(self, frequency, phase, duration):
        return phase + (duration * frequency * 2)

    def copy_defs(self, q):
        """
        For each port that is set to copy from another port, create pulse definitions
        on that port based on the target port's definitions.
        """
        # Identify which ports are in copy mode
        for p, settings in enumerate(self.port_settings):
            port = p + 1
            mode = settings['Mode']
            if mode != 'Copy':
                # Port is not in copy mode
                continue

            target = settings['Sibling']
            # Only copy from ports that have pulses defined to them
            if not self.getValue(f'Port {target} - mode') == 'Define':
                raise ValueError(f'Output port {port} is set to copy from port {target}, '
                                 f'which is either undefined or a copy!')

            amp_shift = self.getValue(f'Port {port} - amplitude scale shift')
            phase_shift = self.getValue(f'Port {port} - phase shift')

            # Copy pulse defs
            idx = 0
            while idx < len(self.pulse_definitions):
                pulse = self.pulse_definitions[idx]
                # Sample pulse definitions do not have a Port value, so they should be ignored
                if ('Sample' not in pulse
                        and pulse['Port'] == target
                        and 'DRAG_idx' not in pulse):
                    # Copy every pulse, but update the output port and apply shifts.
                    p_copy = pulse.copy()
                    p_copy['ID'] = self.get_next_pulse_id()
                    p_copy['Port'] = port
                    p_copy['Phase'] = [phase + phase_shift for phase in pulse['Phase']]
                    p_copy['Amp'] = [amp * amp_shift for amp in pulse['Amp']]
                    self.pulse_definitions.insert(idx + 1, p_copy)

                    # Set up the old target pulse's template on the new port
                    self.setup_template(q, port, p_copy['Carrier'], p_copy['Template_no'])

                    # Step past the newly added pulse
                    idx += 2
                else:
                    idx += 1

    def apply_LUTs(self, q):
        """
        Set up amplitude and frequency/phase LUTs on the board.
        """
        for p in range(8):
            port = p + 1
            for c in range(2):
                # Break apart our freq-phase pairs
                freq_values = []
                phase_values = []
                for (fp1, fp2) in self.fp_matrix[p]:
                    if c == 0:
                        freq = fp1[0]
                        phase = fp1[1]
                    else:
                        freq = fp2[0]
                        phase = fp2[1]

                    freq_values.append(freq)
                    # Make sure phase stays in range
                    phase_sign = np.sign(phase)
                    phase = (abs(phase) % 2) * phase_sign
                    assert -2 <= phase <= 2, 'Phase somehow ended up outside the range [-2,2]!'
                    phase_values.append(phase * np.pi)
                # Feed our values into the tables
                if len(freq_values) > 0 and len(phase_values) > 0:
                    self.add_debug_line(f'q.setup_freq_lut(port={port}, carrier={c+1}, freq={freq_values}, phase={phase_values})')
                    try:
                        q.setup_freq_lut(port, c+1, freq_values, phase_values)
                    except ValueError as err:
                        err_str = err.args[0]
                        if err_str == 'Invalid frequency':
                            raise ValueError(f'The frequency on port {port} is outside '
                                             f'the valid range [0, 2E9] at some point!')
                        if err_str.startswith('frequency_lut can contain at most'):
                            max_num = [int(s) for s in err_str.split() if s.isdigit()][0]
                            raise ValueError(f'There are more than the max number ({max_num}) of '
                                             f'frequency/phase values on port {port}!')
                        raise err
            if len(self.amp_matrix[p]) > 0:
                self.add_debug_line(f'q.setup_scale_lut(port={port}, amp={self.amp_matrix[p]})')
                try:
                    q.setup_scale_lut(port, self.amp_matrix[p])
                except ValueError as err:
                    err_str = err.args[0]
                    if err_str.startswith('scale can contain at most'):
                        max_num = [int(s) for s in err_str.split() if s.isdigit()][0]
                        raise ValueError(f'There are more than the max number ({max_num}) of '
                                         f'amplitude scale values on port {port}!')
                    if err_str.startswith('scale must be in'):
                        raise ValueError(f'The amplitude scale on port {port} '
                                         f'is outside the range [0, 1] at some point!')
                    raise err

    def setup_sequence(self, q):
        """
        Issue commands to the board to set up the pulse sequence defined in the instrument.
        """
        # The time at which the latest emitted pulse ended, for each port
        # Used to determine when to step in the LUTs, to ensure that all parameters are ready
        # before outputting the next pulse
        latest_output = [0] * 8
        self.setup_carriers(q)
        for i in range(self.iterations):
            self.add_debug_line(f'-- Iteration {i} --')
            for pulse in self.pulse_definitions:
                latest_output = self.setup_pulse(i, latest_output, pulse, q)

    def setup_carriers(self, q):
        for p, port_changes in enumerate(self.carrier_changes):
            for i, (t, fp1, fp2) in enumerate(port_changes):
                # Step to the pulse's parameters in the LUTs
                self.go_to_fp(q, t, p+1, (fp1, fp2))
                # If not at the last change, calculate how long we can output until the next change
                if i != len(port_changes) - 1:
                    duration = port_changes[i+1][0] - t - 2e-9
                # The last carrier of the last iteration can run until the iteration's end
                else:
                    duration = (self.measurement_period * self.iterations) - t - 2e-9
                self.add_debug_line(f"q.output_carrier(time={t}, duration={duration}, port={p+1})")
                q.output_carrier(t, duration, p+1)

    def setup_pulse(self, iteration, latest_output, pulse, q):
        """
        Called from setup_sequence. This function handles all the setup of a single pulse definition.
        Returns latest_output, which is the time at which this pulse ends (unless it is a sample pulse,
        in which case the given latest output is returned)
        """
        if 'Sample' in pulse:
            start_base, start_delta = pulse['Time']
            self.add_debug_line(f'q.store(time={self.get_absolute_time(start_base, start_delta, iteration)})')
            q.store(self.get_absolute_time(start_base, start_delta, iteration))
        else:
            port = pulse['Port']
            carrier = pulse['Carrier']
            template_no = pulse['Template_no']
            template_def = self.template_defs[template_no - 1]

            # DRAG pulses have their templates stored in a special way
            if 'DRAG_idx' in pulse:
                template = self.drag_templates[pulse['DRAG_idx']][0]
            else:
                template = self.templates[port - 1][carrier - 1][template_no - 1]

            # Check if template is a long drive. If so, we might need to update its duration
            if 'Base' in template_def:
                (dur_base, dur_delta) = template_def['Base'], template_def['Delta']
                duration = dur_base + dur_delta * iteration
                self.set_long_duration(template_def, template, duration, q)
            else:
                duration = template_def['Duration']

            # Get pulse parameters
            start_base, start_delta = pulse['Time']
            time = self.get_absolute_time(start_base, start_delta, iteration)
            p_amp, p_freq, p_phase = self.get_amp_freq_phase(pulse, iteration)

            # Step to the pulse's parameters in the LUTs
            self.go_to_amp(q, latest_output[port - 1], port, p_amp)

            # Set up the actual pulse command on the board
            self.output_pulse(time, duration, port, template, template_def, q)
            # Store the time at which this pulse ended
            latest_output[port - 1] = time + duration
            # Check that we haven't exceeded trigger period length
            period_end_time = self.get_absolute_time(self.measurement_period, 0, iteration)
            if latest_output[port - 1] >= period_end_time:
                raise ValueError(f'A pulse on port {port} ends after the end '
                                 f'of the trigger period in iteration {iteration + 1}!')
        return latest_output

    def get_amp_freq_phase(self, pulse, iteration):
        """
        Return the amplitude, frequency and phase parameters for a given
        pulse definition in the given iteration step.
        """
        p_amp = pulse['Amp'][iteration]
        p_freq = pulse['Freq'][iteration]
        p_phase = pulse['Phase'][iteration]
        return p_amp, p_freq, p_phase

    def set_long_duration(self, template_def, template, duration, q):
        """
        Give a command to the board to update the given long drive template's duration.
        """
        # If we are using gaussian flanks, the duration must be shortened
        if isinstance(template, tuple):
            long_duration = duration - 2 * template_def['Flank Duration']
            self.add_debug_line(f'update_total_duration({long_duration})')
            template[1].update_total_duration(long_duration)
        else:
            self.add_debug_line(f'update_total_duration({duration})')
            template.update_total_duration(duration)

    def output_pulse(self, time, duration, port, template, template_def, q):
        """
        Called from setup_pulse(), this method handles the calls to the board
        for outputting carrier pulses and envelope pulses for a specific pulse
        definition.
        """
        # If we are doing a long drive with gaussian flanks, we need to output three templates
        if isinstance(template, tuple):
            flank_duration = template_def['Flank Duration']
            # Rise
            rise_template = template[0]
            self.add_debug_line(f'q.output_pulse(time={time}, template={[rise_template]})')
            q.output_pulse(time, [rise_template])
            # Long
            long_template = template[1]
            self.add_debug_line(f'q.output_pulse(time={time + flank_duration}, template={[long_template]})')
            q.output_pulse(time + flank_duration, [long_template])
            # Fall
            fall_template = template[2]
            self.add_debug_line(f'q.output_pulse(time={time + (duration - flank_duration)}, template={[fall_template]})')
            q.output_pulse(time + (duration - flank_duration), [fall_template])
        else:
            self.add_debug_line(f'q.output_pulse(time={time}, template={[template]})')
            q.output_pulse(time, [template])

    def get_absolute_time(self, base, delta, iteration):
        """
        Given a base, a delta and the current iteration index, compute an absolute time value
        that can be used with SimpleQ's methods.
        Every time is also offset by 2 µs, since that is the earliest time a pulse can be sent.
        Return an absolute time based on the input parameters.
        """
        return (self.measurement_period * iteration  # The current period's start time
                + base  # The constant part of the given time
                + delta * iteration)  # The scaling part of the given time

    def go_to_amp(self, q, time, port, amp):
        """
        Perform a number of steps in the board's amplitude scale LUT
        until it is pointing to the given amplitude value.
        """
        p = port - 1

        # Find index of desired value
        for i, a in enumerate(self.amp_matrix[p]):
            if math.isclose(a, amp):
                self.add_debug_line(f"q.select_scale(time={time}, idx={i}, port={port})")
                q.select_scale(time, i, port)

    def go_to_fp(self, q, time, port, fp1fp2):
        """
        Perform a number of steps in the board's frequency/phase LUT
        until it is pointing to the given values.
        """
        p = port - 1

        for i, fpfp in enumerate(self.fp_matrix[p]):
            if self.are_fp_pairs_close(fp1fp2, fpfp):
                self.add_debug_line(f"q.select_frequency(time={time}, idx={i}, port={port})")
                q.select_frequency(time, i, port)

# ▗▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▖ DEBUG ▗▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▖

    def add_debug_line(self, string):
        """
        DEBUG:
        Add the given string as a line to the string that will be written to the log file.
        Also writes the current version of that string to the log file.
        """
        if self.DEBUG_ENABLE:
            if self.INITIAL_TIME is None:
                self.INITIAL_TIME = time.time()
            self.debug_contents += str(time.time() - self.INITIAL_TIME) + ": " + string + '\n'
            # Write what has been accumulated so far
            self.print_lines()

    def print_lines(self):
        """
        DEBUG:
        Write the accumulated debug string to the log file.
        """
        if self.DEBUG_ENABLE:
            # If no file name is specified, default to log.txt
            filename = self.DEBUG_FILE_NAME if self.DEBUG_FILE_NAME != '' else 'log.txt'
            directory = os.path.join(self.USER_DIR, self.DEBUG_PATH)
            Path(directory).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(directory, filename), 'w') as f:
                f.write(self.debug_contents + '\n')

    def print_trigger_sequence(self, q):
        """
        DEBUG:
        Print out the low-level instructions that are set up in SimpleQ.
        """
        if self.DEBUG_ENABLE:
            # If no file name is specified, default to log.txt
            filename = self.DEBUG_FILE_NAME if self.DEBUG_FILE_NAME != '' else 'log.txt'
            directory = os.path.join(self.USER_DIR, self.DEBUG_PATH)
            Path(directory).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(directory, filename), 'w') as f:
                seq = q.seq
                for line in seq:
                    parts = str(line).split(',', 1)
                    f.write(parts[1] + '\n')
