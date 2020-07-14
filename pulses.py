# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for assembling pulse definitions based on ViPS input.
"""

import numpy as np

import input_handling
import utils


def get_next_pulse_id(vips):
    """
    Get a unique pulse ID, for use in pulse definition dictionaries.
    """
    vips.pulse_id_counter += 1
    return vips.pulse_id_counter - 1


def get_all_pulse_defs(vips, q):
    """
    Get the user-defined pulse sequence information for each port.
    Return a list of the pulse definition dictionaries.
    """
    pulse_definitions = []

    # Go through every port definition section one after another
    for port in range(1, vips.N_OUT_PORTS+1):

        settings = vips.port_settings[port - 1]

        # If no pulses are set up on the port, skip to the next
        if settings['Mode'] != 'Define':
            continue

        # Check how many pulses are defined
        n_pulses = int(vips.getValue(f'Pulses for port {port}'))
        # Step through all pulse definitions
        for p_def_idx in range(1, n_pulses + 1):
            pulse_defs = create_pulse_defs(vips, port, p_def_idx, q)
            pulse_definitions.extend(pulse_defs)
    return pulse_definitions


def create_pulse_defs(vips, port, def_idx, q):
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
    template_no = int(vips.getValue(f'Port {port} - def {def_idx} - template'))
    carrier = get_carrier_index(vips.getValue(f'Port {port} - def {def_idx} - sine generator'))

    # Non-DRAG pulses can have their template set up normally
    if carrier != 3:
        try:
            setup_template(vips, q, port, carrier, template_no)
        except ValueError as err:
            if str(err).startswith("No template def"):
                err_msg = f'Pulse definition {def_idx} on port {port} uses an undefined template!'
                raise ValueError(err_msg)
            raise err

    sweep_param = vips.getValue(f'Port {port} - def {def_idx} - Sweep param')
    if sweep_param == 'Amplitude scale':
        amp = get_sweep_values(vips, port, def_idx)
    else:
        amp = vips.getValue(f'Port {port} - def {def_idx} - amp')
        amp = [amp] * vips.iterations
    if sweep_param == 'Carrier frequency':
        freq = get_sweep_values(vips, port, def_idx)
    else:
        freq = vips.getValue(f'Port {port} - def {def_idx} - freq')
        freq = [freq] * vips.iterations
    if sweep_param == 'Phase':
        phase = get_sweep_values(vips, port, def_idx)
    else:
        phase = vips.getValue(f'Port {port} - def {def_idx} - phase')
        phase = [phase] * vips.iterations

    repeat_count = int(vips.getValue(f'Port {port} - def {def_idx} - repeat count'))
    if repeat_count > 1:
        # Get the pulse's duration
        template_def = vips.template_defs[template_no - 1]
        if 'Duration' not in template_def:
            raise ValueError(f'Pulse definition {def_idx} on port {port}: '
                             f'Pulses that use Long drive envelopes cannot be repeated!')
        duration = template_def['Duration']
    else:
        duration = None

    start_times = vips.getValue(f'Port {port} - def {def_idx} - start times')
    start_times = start_times.split(',')
    n_start_times = len(start_times)

    # We define a new pulse for every start time
    pulse_defs = []
    for i, start_time in enumerate(start_times):
        # Parse the original start times and make copies if requested
        if i < n_start_times:
            # Get start time value
            try:
                time = input_handling.parse_number(start_time)
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
                'ID': get_next_pulse_id(vips),
                'Time': time,
                'Port': port,
                'Carrier': carrier,
                'Template_no': template_no,
                'Amp': amp.copy(),
                'Freq': freq.copy(),
                'Phase': phase.copy()})

        # If the pulse is in DRAG mode, we need to calculate some extra parameters
        else:
            pulse_defs.extend(
                calculate_drag(vips, def_idx, time, port, template_no, amp.copy(), freq.copy(), phase.copy(), q))

    return pulse_defs


def get_carrier_index(option):
    """
    Get an integer representing a pulse definition's carrier mode.
    """
    if option == 'DRAG':
        return 3
    if option == 'None':
        return 0
    return int(option)


def calculate_drag(vips, def_idx, time, port, template_no, amp, freq, phase, q):
    """
    Creates four DRAG pulses based on a pulse definition set to DRAG mode.
    This will also result in the creation of four new templates on the board, which will be
    stored in vips.drag_templates, at an index saved in each pulse definition's 'DRAG_idx' key.
    Returns a list of the four pulse definitions.
    """

    sibling_port = int(vips.getValue(f'Port {port} - def {def_idx} - DRAG sibling port'))
    times, points = utils.template_def_to_points(vips, vips.template_defs[template_no - 1], 0)
    scale = vips.getValue(f'Port {port} - def {def_idx} - DRAG scale')
    detuning = vips.getValue(f'Port {port} - def {def_idx} - DRAG detuning frequency')
    phase_shift = vips.getValue(f'Port {port} - def {def_idx} - DRAG phase shift')

    if (template_no, scale, detuning) not in vips.drag_parameters:

        beta = scale * vips.sampling_freq

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
        vips.lgr.add_line(f'q.setup_template(port={port}, points={re_points}, carrier=1, use_scale=True)')
        vips.lgr.add_line(f'q.setup_template(port={port}, points={im_points}, carrier=2, use_scale=True)')
        vips.lgr.add_line(f'q.setup_template(port={sibling_port}, points={re_points}, carrier=1, use_scale=True)')
        vips.lgr.add_line(f'q.setup_template(port={sibling_port}, points={im_points}, carrier=2, use_scale=True)')
        base_re_template = q.setup_template(port, re_points, 1, True)
        base_im_template = q.setup_template(port, im_points, 2, True)
        sibl_re_template = q.setup_template(sibling_port, re_points, 1, True)
        sibl_im_template = q.setup_template(sibling_port, im_points, 2, True)
        vips.drag_templates.append((base_re_template, re_points))
        base_re_idx = len(vips.drag_templates) - 1
        vips.drag_templates.append((base_im_template, im_points))
        base_im_idx = len(vips.drag_templates) - 1
        vips.drag_templates.append((sibl_re_template, re_points))
        sibl_re_idx = len(vips.drag_templates) - 1
        vips.drag_templates.append((sibl_im_template, im_points))
        sibl_im_idx = len(vips.drag_templates) - 1

    # We've already made the necessary templates, find their index
    else:
        match_idx = vips.drag_parameters.index((template_no, scale, detuning))

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
            'ID': get_next_pulse_id(vips),
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


def setup_template(vips, q, port, carrier, template_no):
    """
    Set up the specified template on the specified port on the board.
    Store the template globally in the format given by Vivace.
    """
    template_def = vips.template_defs[template_no - 1]
    # Check that the given template number has a definition
    if len(template_def) == 0:
        raise ValueError("No template def found!")
    # If the requested template has not already been set up for this port, do it.
    if vips.templates[port - 1][carrier - 1][template_no - 1] is None:
        try:
            # Only long drives have the 'Base' key
            if 'Base' in template_def:
                initial_length = template_def['Base']
                # Set up gaussian rise and fall templates if defined.
                if 'Flank Duration' in template_def:
                    initial_length -= 2 * template_def['Flank Duration']
                    vips.lgr.add_line(f'q.setup_template(port={port}, points={template_def["Rise Points"]}, carrier={carrier}, use_scale=True)')
                    rise_template = q.setup_template(port, template_def['Rise Points'], carrier, use_scale=True)
                    vips.lgr.add_line(f'q.setup_template(port={port}, points={template_def["Fall Points"]}, carrier={carrier}, use_scale=True)')
                    fall_template = q.setup_template(port, template_def['Fall Points'], carrier, use_scale=True)
                vips.lgr.add_line(f'q.setup_long_drive(port={port}, carrier={carrier}, duration={initial_length}, use_scale=True)')
                try:
                    long_template = q.setup_long_drive(port,
                                                       carrier,
                                                       initial_length,
                                                       use_scale=True)
                except ValueError as err:
                    if err.args[0].startswith('valid carriers'):
                        raise ValueError('Long drive envelopes have to be on either sine generator 1 or 2!')
                if 'Flank Duration' in template_def:
                    vips.templates[port - 1][carrier - 1][template_no - 1] = (rise_template, long_template, fall_template)
                else:
                    vips.templates[port - 1][carrier - 1][template_no - 1] = long_template
            else:
                vips.lgr.add_line(f'q.setup_template(port={port}, points={template_def["Points"]}, carrier={carrier}, use_scale=True)')
                vips.templates[port - 1][carrier - 1][template_no - 1] = q.setup_template(port,
                                                                             template_def['Points'],
                                                                             carrier,
                                                                             use_scale=True)
        except RuntimeError as error:
            if error.args[0].startswith('Not enough templates on output'):
                raise RuntimeError(f'There are more than 16 templates in use on port {port}!\n '
                                   '(Templates longer than 1024 ns are split into multiple, '
                                   'unless they are of type "Long drive")')


def get_sample_pulses(vips, q):
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
    for port in range(1, vips.N_IN_PORTS+1):
        use_port = vips.getValue(f'Sampling on port {port}')
        if use_port:
            sampling_ports.append(port)
    if len(sampling_ports) == 0:
        raise ValueError('Sampling not set up on any port!')

    duration = vips.getValue(f'Sampling - duration')
    # The board can only sample for 4096 ns at a time, so if the user wants longer, we need to split up the calls.
    if duration > 4096e-9:
        raise ValueError('Sampling duration must be in [0.0, 4096.0] ns')

    vips.lgr.add_line(f'q.set_store_duration({duration})')
    q.set_store_duration(duration)

    # Save the ports we want to sample on
    vips.lgr.add_line(f'q.set_store_ports({sampling_ports})')
    q.set_store_ports(sampling_ports)
    vips.store_ports = sampling_ports

    # Get times and duration
    start_times_string = vips.getValue(f'Sampling - start times')
    start_times = start_times_string.split(',')
    vips.samples_per_iteration = len(start_times)

    # Store the sample pulse's defining parameters
    for start_time in start_times:
        try:
            time = input_handling.parse_number(start_time)
        except ValueError:
            err_msg = f'Invalid start time definition for sampling!'
            raise ValueError(err_msg)
        # Get a unique pulse def id
        p_id = get_next_pulse_id(vips)

        sample_definitions.append({
            'ID': p_id,
            'Time': time,
            'Sample': True})

    return sample_definitions


def get_sweep_values(vips, port, def_idx):
    """
    Calculate and return a list of parameter values to sweep over based on the given pulse's sweep settings.
    """
    sweep_format = vips.getValue(f'Port {port} - def {def_idx} - Sweep format')
    # Custom is a special case, we just get the values directly
    if sweep_format == 'Custom':
        step_values = vips.getValue(f'Port {port} - def {def_idx} - Sweep custom steps')
        string_list = step_values.split(',')
        if len(string_list) != vips.iterations:
            raise ValueError(f'The number of custom values for pulse definition '
                             f'{def_idx} on port {port} does not match the number of iterations!')
        values = []
        for string in string_list:
            values.append(float(string))
        return values
    # For linear, we need to calculate the full list of values
    if sweep_format == 'Linear: Start-End':
        interval_start = vips.getValue(f'Port {port} - def {def_idx} - Sweep linear start')
        interval_end = vips.getValue(f'Port {port} - def {def_idx} - Sweep linear end')
    else:  # Center-span
        center = vips.getValue(f'Port {port} - def {def_idx} - Sweep linear center')
        span = vips.getValue(f'Port {port} - def {def_idx} - Sweep linear span')
        interval_start = center - (span / 2)
        interval_end = center + (span / 2)

    return list(np.linspace(interval_start, interval_end, vips.iterations))
