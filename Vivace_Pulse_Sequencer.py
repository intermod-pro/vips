# Authored by Johan Blomberg and Gustav Grännsjö, 2020

import os

import numpy as np

from BaseDriver import LabberDriver, Error

from vivace import pulsed
import input_handling
import logger
import utils
import previews
import pulses
import templates
import luts
import template_matching
import scheduling


class Driver(LabberDriver):
    """
    Welcome to the instrument driver for the Vivace Pulse Sequencer (ViPS).
    This class handles the core of the instrument's behaviour, and delegates
    calls to other files that are responsible for specific tasks.
    An instance of this class is passed around to these methods in order to fill
    it with data extracted from different parts of the instrument.
    """

    # In order to use the driver with actual hardware, DRY_RUN needs to be False
    DRY_RUN = True

    N_IN_PORTS = 8
    N_OUT_PORTS = 8

    # Version numbers, will be fetched from Vivace and the ViPS definition
    vips_ver = None
    vivace_fw_ver = None
    vivace_server_ver = None
    vivace_api_ver = None
    
    # Logger for debug purposes
    lgr = logger.Logger()

    def __init__(self, dInstrCfg=None, dComCfg=None, dValues=None, dOption=None,
                 dPrefs={}, queueIn=None, queueOut=None, logger=None):
        LabberDriver.__init__(self, dInstrCfg, dComCfg, dValues, dOption,
                              dPrefs, queueIn, queueOut, logger)

        # IP address used to connect to board
        self.address = self.getAddress()

        # Get the relevant version numbers
        self.fetch_version_numbers()

        self.sampling_freq = None
        self.averages = None
        self.measurement_period = None
        self.iterations = None
        self.templates = [[[None for _ in range(16)] for _ in range(2)] for _ in range(self.N_OUT_PORTS)]
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
        self.template_matchings = None
        self.match_results = None

        # Sample ports
        self.store_ports = None

        # Measurement output
        self.time_array = None
        self.sampling_results = None

        # This list is used to keep track of the specific options used when getting traces in Labber
        self.previously_outputted_trace_configs = []

        self.lgr.new_log = True

    def reset_instrument(self):
        """
        Reinitialise the driver's state. Should be equivalent to restarting the instrument.
        """
        self.averages = None
        self.measurement_period = None
        self.iterations = None
        self.templates = [[[None for _ in range(16)] for _ in range(2)] for _ in range(self.N_OUT_PORTS)]
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
        self.template_matchings = None
        self.match_results = None

        # Sample ports
        self.store_ports = None

        # Measurement output
        self.time_array = None
        self.sampling_results = None

        self.previously_outputted_trace_configs = []

        # Prepare to start a new log file
        self.lgr.new_log = True

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
            if input_handling.is_value_new(self, quant, value):
                self.reset_instrument()

        # Version numbers should be kept constant
        if 'vips_version' in set_commands:
            return self.vips_ver

        if 'vivace_fw_version' in set_commands:
            return self.vivace_fw_ver

        if 'vivace_server_version' in set_commands:
            return self.vivace_server_ver

        if 'vivace_api_version' in set_commands:
            return self.vivace_api_ver

        return input_handling.handle_input(quant, value)

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
            self.vips_ver = version

        # Get Vivace's version numbers
        with pulsed.Pulsed(dry_run=self.DRY_RUN, address=self.address) as q:
            self.vivace_api_ver = '1.0.2'  # TODO fetch from Vivace

            try:
                fw_ver = q._rflockin.read_register(24)
                server_ver = q._rflockin.get_server_version()
            except AttributeError:
                fw_ver = 'Could not connect to Vivace :('
                server_ver = 'Could not connect to Vivace :('
            self.vivace_fw_ver = fw_ver
            self.vivace_server_ver = server_ver

    def performGetValue(self, quant, options={}):
        """
        Perform the Get Value instrument operation.
        Responsible for starting measurements and fetching and formatting the resulting time traces.
        Return the requested time trace in the form of a Labber Trace Dict.
        """

        if quant.get_cmd == 'get_trace':

            # If we don't have any results, get some
            if self.sampling_results is None:
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
            self.lgr.add_line(str(circumstance))
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
        if quant.get_cmd == 'get_match':
            # If we don't have any matching data, get some
            if self.match_results is None:
                self.perform_measurement()

            # TODO

            return None

        if quant.get_cmd == 'template_preview':
            return previews.get_template_preview(self, quant)
        if quant.get_cmd == 'sequence_preview':
            return previews.get_sequence_preview(self, quant)

        return quant.getValue()

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

        measurement = self.sampling_results[window_idx][output_idx]
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
            luts.apply_LUTs(self, q)

            # Set up the full pulse on the board
            scheduling.setup_sequence(self, q)

            # Schedule template matching
            scheduling.setup_template_matches(self, q)

            # Start measuring
            total_time = self.measurement_period * (self.iterations + 1)
            self.lgr.add_line('q.perform_measurement()')
            output = q.perform_measurement(total_time, 1, self.averages)
            if not q.dry_run:
                # Store the results
                (t_array, result) = output
                self.time_array = list(t_array)
                self.sampling_results = result
                self.match_results = self.get_template_matching_results(q)
                assert False, self.match_results
            else:
                self.sampling_results = 'Dummy result'

    def get_template_matching_results(self, q):
        """
        Take the stored template matching information and feed it to Vivace to
        request matching results.
        """
        matchings = []
        for m in self.template_matchings:
            matchings.extend([m[1], m[2]])

        return q.get_template_matching_data(matchings)

    def setup_instrument(self, q):
        """
        Fetch all user-given data from the instrument, and process it appropriately.
        Store pulse definitions, envelope templates, and LUTs, for use by other methods.
        """
        # Get debug information
        self.get_debug_settings()

        self.sampling_freq = q.sampling_freq
        # Get some general parameters such as no. of averages, trigger period etc.
        self.get_general_settings()
        # Get template definitions
        self.template_defs = templates.get_template_defs(self)
        # Port settings
        self.port_settings = self.get_port_settings()
        # Set DC biases for all ports
        self.set_dc_biases(q)
        # Pulse definitions
        self.pulse_definitions = pulses.get_all_pulse_defs(self, q)
        # Sampling
        sample_definitions = pulses.get_sample_pulses(self, q)
        self.pulse_definitions.extend(sample_definitions)
        # Copy pulse definitions on specified ports
        self.copy_defs(q)
        # Sort our definitions chronologically
        self.pulse_definitions = sorted(self.pulse_definitions,
                                        key=lambda x: x['Time'][0] + x['Time'][1])
        self.validate_pulse_definitions()
        # Get the values that will go in the LUTs
        self.amp_matrix, self.fp_matrix, self.carrier_changes = luts.get_LUT_values(self)
        # Get template matching data
        self.template_matchings = template_matching.get_template_matching_definitions(self, q)

    def get_debug_settings(self):
        """
        Fetch the user-specified debug-related settings and pass them on to the logger.
        """
        self.lgr.enable = self.getValue('Enable Vivace call logging')

        if self.lgr.enable:
            log_name = self.getValue('Log file name')
            for c in '\/:*?"<>|':
                if c in log_name:
                    raise ValueError(f'The log file name contains the illegal character {c}!')

            # If no file name is specified, default to log
            self.lgr.file_name = log_name if log_name != '' else 'log'
            self.lgr.working_file_name = self.lgr.file_name

            self.lgr.overwrite = self.getValue('Overwrite previous log')

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

    def get_port_settings(self):
        """
        Get each port's settings.
        Port settings are represented by an array of dictionaries, each representing a single output port's settings.
            The 'Mode' key contains the port's mode (Define, Disabled, Copy)
            If a port is in copy mode, also save the port it is copying from in a 'Sibling' key.
        Return a list of these dictionaries.
        """
        port_settings = [{} for _ in range(self.N_OUT_PORTS)]

        # Get the mode for each port
        for port in range(1, self.N_OUT_PORTS+1):
            p = port - 1
            mode = self.getValue(f'Port {port} - mode')
            port_settings[p]['Mode'] = mode
            if mode == 'Copy':
                port_settings[p]['Sibling'] = int(self.getValue(f'Port {port} - copy sequence from'))

        return port_settings

    def set_dc_biases(self, q):
        if not q.dry_run:
            for port in range(1, self.N_OUT_PORTS+1):
                bias = self.getValue(f'Port {port} - DC bias')
                bias = bias / 1.25
                self.lgr.add_line(f'q._rflockin.set_bias_dac(port={port}, bias={bias})')
                q._rflockin.set_bias_dac(port, bias)

    def validate_pulse_definitions(self):
        """
        Ensure that pulse definitions on different carrier generators of the same port
        interact in a safe way. Pulses can only overlap if they have the same start and end time,
        and their combined amplitude cannot exceed 1.
        """
        for p in range(self.N_OUT_PORTS):
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
                    curr_start = utils.get_absolute_time(self, start[0], start[1], it)
                    temp_def = self.template_defs[pulse['Template_no']-1]
                    curr_duration = self.get_template_def_duration(temp_def, it)
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

    def get_template_def_duration(self, tempdef, iteration):
        """
        Compute and return the total duration of a template definition for a given iteration.
        """
        if 'Base' in tempdef:  # Long drive
            return tempdef['Base'] + tempdef['Delta'] * iteration
        else:
            return tempdef['Duration']

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
                    p_copy['ID'] = pulses.get_next_pulse_id(self)
                    p_copy['Port'] = port
                    p_copy['Phase'] = [phase + phase_shift for phase in pulse['Phase']]
                    p_copy['Amp'] = [amp * amp_shift for amp in pulse['Amp']]
                    self.pulse_definitions.insert(idx + 1, p_copy)

                    # Set up the old target pulse's template on the new port
                    pulses.setup_template(self, q, port, p_copy['Carrier'], p_copy['Template_no'])

                    # Step past the newly added pulse
                    idx += 2
                else:
                    idx += 1
