# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for setting up template matching.
"""

import numpy as np

import utils


def get_template_matching_definitions(vips, q):
    """
    Fetch template matching data from the instrument and use it to
    set up template matches in Vivace. Return a list of tuples containing each
    match's time, I port match TrigEvent and Q port match TrigEvent.
    """
    n_matches = int(vips.getValue('Number of matches'))

    matchings = []

    for m in range(1, n_matches+1):
        (sample_i_port,
         sample_q_port,
         pulse_i_port,
         pulse_q_port,
         i_is_base) = get_port_information(vips, m)
        pulse_base_port = pulse_i_port if i_is_base else pulse_q_port
        pulse_copy_port = pulse_q_port if i_is_base else pulse_i_port

        matching_start = vips.getValue(f'Template matching {m} - matching start time')
        padding_length = ((matching_start * 1e9) % 2) / 1e9
        matching_start = matching_start - padding_length
        match_duration = vips.getValue(f'Template matching {m} - matching duration')

        pulse_start = vips.getValue(f'Template matching {m} - pulse start time')
        p_freq = vips.getValue(f'Template matching {m} - pulse frequency')
        p_phase = vips.getValue(f'Template matching {m} - pulse phase')

        window_duration = vips.getValue('Sampling - duration')

        # Matching can only happen within sampling windows
        for i in range(vips.iterations):
            within_window = False
            for pulse in vips.pulse_definitions:
                if 'Sample' in pulse:
                    abs_match_start = utils.get_absolute_time(vips, matching_start, 0, i)
                    abs_window_start = utils.get_absolute_time(vips, pulse['Time'][0], pulse['Time'][1], i)

                    if (abs_match_start >= abs_window_start
                            and abs_match_start + match_duration <= abs_window_start + window_duration):
                        within_window = True
                        break

            if not within_window:
                # Matching was not within any sampling window
                err_msg = f'Template matching {m}: The template matching needs to occur within a sampling window! ' \
                          f'It is first outside of a sampling window on iteration {i+1}.'
                raise ValueError(err_msg)

        # Find a pulse to phase sync with, based on the given frequency value
        adjusted_phase = None

        for pulse in vips.pulse_definitions:
            if 'Sample' in pulse or pulse['Port'] != pulse_base_port:
                continue

            if pulse['Freq'][0] == p_freq:
                # Calculate phase of matching template based on the reference pulse
                reference_time = pulse['Time'][0]
                adjusted_phase = utils.phase_sync(p_freq, p_phase, pulse_start - reference_time)
                break

        # No pulse of matching frequency found
        if adjusted_phase is None:
            err_msg = f'Template matching {m}: The given frequency value does not match that of ' \
                      f'any existing pulse definition!'
            raise ValueError(err_msg)

        # Construct matching template
        envelope = np.ones(round(match_duration * vips.sampling_freq))
        pad_points = int(padding_length * 4e9)
        envelope = np.concatenate((np.zeros(pad_points), envelope))

        # Construct carriers to modulate template with
        end_point = (len(envelope) - 1) / vips.sampling_freq
        x = np.linspace(0, end_point, len(envelope))
        carrier_base = np.cos(2 * np.pi * p_freq * x + np.pi * adjusted_phase)
        # Only fetch the phase shift value if the copy port is used
        phase_shift = vips.getValue(f'Port {pulse_copy_port} - phase shift') if pulse_copy_port != 0 else 0
        carrier_copy = np.cos(2 * np.pi * p_freq * x + np.pi * (adjusted_phase + phase_shift))

        # Modulate matching template with carrier
        m_template_i = envelope * carrier_base if i_is_base else envelope * carrier_copy
        m_template_q = envelope * carrier_copy if i_is_base else envelope * carrier_base

        vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_i_port}, template1={m_template_i})')
        matching_i = q.setup_template_matching_pair(sample_i_port, m_template_i)
        if pulse_copy_port != 0:
            vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_q_port}, template1={m_template_q})')
            matching_q = q.setup_template_matching_pair(sample_q_port, m_template_q)
            matchings.append((matching_start, matching_i, matching_q, match_duration))
        else:
            matchings.append((matching_start, matching_i, None, match_duration))

    return matchings


def get_port_information(vips, matching_no):
    """
    Get information on what input and output ports are involved in a match,
    and check that everything is set up correctly.
    Return the I and Q input ports (on which the matching is performed), the
    pulse I and Q ports (where the pulse to be matched is set up) and a flag
    indicating whether the pulse's I port is in Define or Copy mode.
    """
    sample_i_port = int(vips.getValue(f'Template matching {matching_no} - sampling I port'))
    sample_q_port = vips.getValue(f'Template matching {matching_no} - sampling Q port')
    if sample_q_port == 'None':
        sample_q_port = 0
    else:
        sample_q_port = int(sample_q_port)

    # Matching can only happen on ports with sampling activated
    if sample_i_port not in vips.store_ports or sample_q_port not in [0, *vips.store_ports]:
        err_msg = f'Template matching {matching_no}: Sampling needs to be enabled on the ports set as sampling ports!'
        raise ValueError(err_msg)

    pulse_i_port = int(vips.getValue(f'Template matching {matching_no} - pulse I port'))
    pulse_q_port = vips.getValue(f'Template matching {matching_no} - pulse Q port')
    if pulse_q_port == 'None':
        pulse_q_port = 0
        # If the sampling Q port is set to None, the output Q port has to match
        if sample_q_port != 0:
            err_msg = f'Template matching {matching_no}: The sampling Q port is defined, but the pulse Q port is set ' \
                      f'to None. To template match on port {sample_q_port}, ViPS needs to know where the readout ' \
                      f'pulse is outputted.'
            raise ValueError(err_msg)

    else:
        pulse_q_port = int(pulse_q_port)
        # Pulse Q is defined, so sampling Q port also has to be
        if sample_q_port == 0:
            err_msg = f'Template matching {matching_no}: The pulse Q port is defined, but the sampling Q port is set ' \
                      f'to None. If you only wish to template match on one port, there is no need to define two ' \
                      f'pulse ports.'
            raise ValueError(err_msg)

    # The port pair needs to be well-defined
    if pulse_i_port == pulse_q_port:
        err_msg = f'Template matching {matching_no}: pulse I and Q ports cannot have the same port number!'
        raise ValueError(err_msg)

    if sample_i_port == sample_q_port:
        err_msg = f'Template matching {matching_no}: sampling I and Q ports cannot have the same port number!'
        raise ValueError(err_msg)

    if pulse_q_port != 0:
        if vips.port_settings[pulse_i_port-1]['Mode'] == 'Copy':
            i_is_base = False
            if vips.port_settings[pulse_i_port-1]['Sibling'] != pulse_q_port:
                err_msg = f'Template matching {matching_no}: The pulse I port is not set up as a copy of the Q port! ' \
                          f'The I port needs to be a copy of the Q port, or vice versa.'
                raise ValueError(err_msg)

        elif vips.port_settings[pulse_q_port-1]['Mode'] == 'Copy':
            i_is_base = True
            if vips.port_settings[pulse_q_port-1]['Sibling'] != pulse_i_port:
                err_msg = f'Template matching {matching_no}: The pulse Q port is not set up as a copy of the I port! ' \
                          f'The Q port needs to be a copy of the I port, or vice versa.'
                raise ValueError(err_msg)

        else:
            err_msg = f'Template matching {matching_no}: Neither the I nor Q pulse port is defined ' \
                      f'as a copy of the other!'
            raise ValueError(err_msg)

    # Q is set to None
    else:
        if vips.port_settings[pulse_i_port-1]['Mode'] == 'Disabled':
            err_msg = f'Template matching {matching_no}: The pulse I port is set to Disabled!'
            raise ValueError(err_msg)

        i_is_base = True

    return sample_i_port, sample_q_port, pulse_i_port, pulse_q_port, i_is_base
