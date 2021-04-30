# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for assembling pulse definitions based on ViPS input.
"""

import numpy as np

import input_handling
import utils
from templates import TemplateIdentifier


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

    cond_on = vips.getValue(f'Port {port} - def {def_idx} - Condition comparator')
    cond1 = vips.getValue(f'Port {port} - def {def_idx} - Template matching condition 1')
    cond1 = utils.combo_to_int(cond1)
    cond1_quad = vips.getValue(f'Port {port} - def {def_idx} - Template matching condition 1 quadrature')
    cond2 = vips.getValue(f'Port {port} - def {def_idx} - Template matching condition 2')
    cond2 = utils.combo_to_int(cond2)
    cond2_quad = vips.getValue(f'Port {port} - def {def_idx} - Template matching condition 2 quadrature')

    if cond1 != 0 and cond1 == cond2:
        raise ValueError(f'Pulse {def_idx} on port {port}: both output conditions are set '
                         f'to the same value! Please set another value for the second condition,'
                         f'or disable it.')

    template_identifier = TemplateIdentifier(port,
                                             carrier,
                                             template_no,
                                             cond_on,
                                             cond1, cond2,
                                             cond1_quad, cond2_quad)

    # Non-DRAG pulses can have their template set up normally
    if carrier != 3:
        try:
            setup_template(vips, q, template_identifier)
        except ValueError as err:
            if str(err).startswith('No template def'):
                raise ValueError(f'Pulse definition {def_idx} on port {port} uses an undefined template!')
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
                time = input_handling.compute_time_string(vips, start_time)
            except ValueError as err:
                raise ValueError(f'Invalid start time definition for port {port}, definition {def_idx}:\n{err}')

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
                'Template_identifier': template_identifier,
                'Amp': amp.copy(),
                'Freq': freq.copy(),
                'Phase': phase.copy()})

        # If the pulse is in DRAG mode, we need to calculate some extra parameters
        else:
            pulse_defs.extend(
                calculate_drag(vips, def_idx, time, port, template_no, template_identifier, amp.copy(), freq.copy(), phase.copy(), q))

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


def calculate_drag(vips, def_idx, time, port, template_no, template_identifier, amp, freq, phase, q):
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
    sibl_amp_multiplier = vips.getValue(f'Port {port} - def {def_idx} - DRAG amplitude scale multiplier')
    # Extract conditional information from base template identifier
    condition_on = template_identifier.cond_on
    cond1 = template_identifier.cond1
    cond1_quad = template_identifier.cond1_quad
    cond2 = template_identifier.cond2
    cond2_quad = template_identifier.cond2_quad

    if (template_no, template_identifier, scale, detuning) not in vips.drag_parameters:
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
        try:
            base_re_template = q.setup_template(port, 0, re_points)
            base_im_template = q.setup_template(port, 1, im_points)
            sibl_re_template = q.setup_template(sibling_port, 0, re_points)
            sibl_im_template = q.setup_template(sibling_port, 1, im_points)
        except RuntimeError as error:
            if error.args[0].startswith('Not enough templates on output'):
                raise RuntimeError('There are more than 8 templates in use on a single carrier'
                                   f'on port {port} or {sibling_port}!\n The limit was exceeded'
                                   'while setting up a DRAG pulse on these ports.'
                                   '(Templates longer than 1024 ns are split into multiple, '
                                   'unless they are of type "Long drive")')

        # We add 1000 to the base index to separate it from a normal definition index
        param_idx = len(vips.drag_parameters)
        base_re_idx = vips.DRAG_INDEX_OFFSET + param_idx * 4 + 0
        base_im_idx = vips.DRAG_INDEX_OFFSET + param_idx * 4 + 1
        sibl_re_idx = vips.DRAG_INDEX_OFFSET + param_idx * 4 + 2
        sibl_im_idx = vips.DRAG_INDEX_OFFSET + param_idx * 4 + 3

        # Prepare template identifiers to store the four templates under
        base_re_ti = TemplateIdentifier(port,         1, base_re_idx, condition_on, cond1, cond2, cond1_quad, cond2_quad)
        base_im_ti = TemplateIdentifier(port,         2, base_im_idx, condition_on, cond1, cond2, cond1_quad, cond2_quad)
        sibl_re_ti = TemplateIdentifier(sibling_port, 1, sibl_re_idx, condition_on, cond1, cond2, cond1_quad, cond2_quad)
        sibl_im_ti = TemplateIdentifier(sibling_port, 2, sibl_im_idx, condition_on, cond1, cond2, cond1_quad, cond2_quad)
        vips.templates[base_re_ti] = base_re_template
        vips.templates[base_im_ti] = base_im_template
        vips.templates[sibl_re_ti] = sibl_re_template
        vips.templates[sibl_im_ti] = sibl_im_template

        # Also store the points that make up the templates, for preview purposes
        vips.drag_templates.append(re_points)
        vips.drag_templates.append(im_points)
        vips.drag_templates.append(re_points)
        vips.drag_templates.append(im_points)

        # Add these parameters to the list
        vips.drag_parameters.append((template_no, template_identifier, scale, detuning))

    # We've already made the necessary templates, find their index
    else:
        match_idx = vips.drag_parameters.index((template_no, template_identifier, scale, detuning))

        # We add 1000 to the base index to separate it from a normal definition index
        base_re_idx = vips.DRAG_INDEX_OFFSET + match_idx * 4 + 0
        base_im_idx = vips.DRAG_INDEX_OFFSET + match_idx * 4 + 1
        sibl_re_idx = vips.DRAG_INDEX_OFFSET + match_idx * 4 + 2
        sibl_im_idx = vips.DRAG_INDEX_OFFSET + match_idx * 4 + 3

    # Create four pulse defs
    pulse_defs = []
    for i in range(4):
        d_port = port if i in (0, 1) else sibling_port
        d_carrier = 1 if i in (0, 2) else 2

        # Phase offset, ampl scale and template index is different between every definition
        if i == 0:
            d_idx = base_re_idx
            # Cosine, so we don't need to shift the carrier
            d_phase = phase.copy()
            # Don't scale amplitude on the base port
            amp_multiplier = 1
        elif i == 1:
            d_idx = base_im_idx
            # Sine, i.e. a negative pi/2 offset
            d_phase = [p - 0.5 for p in phase.copy()]
            amp_multiplier = 1
        elif i == 2:
            d_idx = sibl_re_idx
            # Cosine
            d_phase = [p + phase_shift for p in phase.copy()]
            # Amplitude on sibling ports may be rescaled by user
            amp_multiplier = sibl_amp_multiplier
        else:
            d_idx = sibl_im_idx
            # Sine
            d_phase = [p - 0.5 + phase_shift for p in phase.copy()]
            amp_multiplier = sibl_amp_multiplier

        # Recreate the template identifier used before
        template_ident = TemplateIdentifier(d_port, d_carrier, d_idx, condition_on, cond1, cond2, cond1_quad, cond2_quad)
        pulse_defs.append({
            'ID': get_next_pulse_id(vips),
            'Time': time,
            'Port': d_port,
            'Carrier': d_carrier,
            'Template_no': template_no,
            'Template_identifier': template_ident,
            'DRAG_idx': d_idx,
            'Amp': [a * amp_multiplier for a in amp.copy()],
            'Freq': freq.copy(),
            'Phase': d_phase}
        )

    return pulse_defs


def setup_template(vips, q, template_identifier):
    """
    Set up the specified template on the specified port on the board.
    Store the template globally in the format given by Vivace.
    """
    template_no = template_identifier.def_idx
    template_def = vips.template_defs[template_no - 1]
    # Check that the given template number has a definition
    if len(template_def) == 0:
        raise ValueError('No template def found!')
    # If the requested template has not already been set up for this port, do it.
    if template_identifier not in vips.templates:
        port = template_identifier.port
        carrier = template_identifier.carrier
        try:
            # Only long drives have the 'Base' key
            if 'Base' in template_def:
                initial_length = template_def['Base']
                # Set up gaussian rise and fall templates if defined.
                if 'Flank Duration' in template_def:
                    initial_length -= 2 * template_def['Flank Duration']
                    vips.lgr.add_line(f'q.setup_template(port={port}, points={template_def["Rise Points"]}, carrier={carrier}, use_scale=True)')
                    group = max(0, carrier-1)
                    rise_template = q.setup_template(port, group, template_def['Rise Points'], envelope=carrier > 0)
                    vips.lgr.add_line(f'q.setup_template(port={port}, points={template_def["Fall Points"]}, carrier={carrier}, use_scale=True)')
                    fall_template = q.setup_template(port, group, template_def['Fall Points'], envelope=carrier > 0)
                vips.lgr.add_line(f'q.setup_long_drive(port={port}, carrier={carrier}, duration={initial_length}, use_scale=True)')
                try:
                    group = max(0, carrier-1)
                    long_template = q.setup_long_drive(port, carrier-1, initial_length)
                except ValueError as err:
                    if err.args[0].startswith('valid carriers'):
                        raise ValueError('Long drive envelopes have to be on either sine generator 1 or 2!')
                if 'Flank Duration' in template_def:
                    vips.templates[template_identifier] = (rise_template, long_template, fall_template)
                else:
                    vips.templates[template_identifier] = long_template
            else:
                vips.lgr.add_line(f'q.setup_template(port={port}, points={template_def["Points"]}, carrier={carrier}, use_scale=True)')
                group = max(0, carrier-1)
                vips.templates[template_identifier] = q.setup_template(port, group, template_def['Points'], envelope=carrier > 0)
        except RuntimeError as error:
            if error.args[0].startswith('Not enough templates on output'):
                raise RuntimeError(f'There are more than 8 templates in use on carrier {carrier}'
                                   f'on port {port}!\n '
                                   '(Templates longer than 1024 ns are split into multiple, '
                                   'unless they are of type "Long drive")')


def get_sample_windows(vips, q):
    """
    Set up the user-defined sample windows on the board.
    These are stored in a dictionary format, with the following entries:
        ID: The window's pulse ID.
        Time: The window's starting time, given as the tuple (base, delta).
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

    # Save the ports we want to sample on
    vips.lgr.add_line(f'q.set_store_ports({sampling_ports})')
    q.set_store_ports(sampling_ports)
    vips.sampling_ports = sampling_ports

    vips.lgr.add_line(f'q.set_store_duration({duration})')
    q.set_store_duration(duration)
    vips.sampling_duration = duration

    # Get times and duration
    start_times_string = vips.getValue(f'Sampling - start times')
    start_times = start_times_string.split(',')
    vips.samples_per_iteration = len(start_times)

    # Store the sample pulse's defining parameters
    for start_time in start_times:
        try:
            time = input_handling.compute_time_string(vips, start_time)
        except ValueError as err:
            raise ValueError(f'Invalid start time definition for sampling:\n{err}')

        # Sample window has to fit within trigger period
        if time[0] + (vips.iterations - 1) * time[1] + duration > vips.trigger_period:
            raise ValueError('Sampling duration cannot exceed the length of the trigger period!')

        # Get a unique pulse def id
        p_id = get_next_pulse_id(vips)

        sample_definitions.append({
            'ID': p_id,
            'Time': time})

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
