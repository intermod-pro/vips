# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for setting up template matching.
"""

import numpy as np

import templates
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
        (sample_port_1,
         sample_port_2,
         pulse_port_1,
         pulse_port_2) = get_port_information(vips, m)

        envelope1 = vips.getValue(f'Template matching {m} - template 1')
        #envelope2 = vips.getValue(f'Template matching {m} - template 2')

        matching_start = vips.getValue(f'Template matching {m} - matching start time')
        padding_length = ((matching_start * 1e9) % 2) / 1e9
        matching_start = matching_start - padding_length
        match_duration = vips.getValue(f'Template matching {m} - matching duration')

        pulse_start = vips.getValue(f'Template matching {m} - pulse start time')
        p_freq = vips.getValue(f'Template matching {m} - pulse frequency')
        p_phase = vips.getValue(f'Template matching {m} - pulse phase')

        q_relative_phase_offset = vips.getValue(f'Template matching {m} - Q port phase adjustment')
        p1_amp_multiplier = vips.getValue(f'Template matching {m} - I port amplitude scale multiplier')
        p2_amp_multiplier = vips.getValue(f'Template matching {m} - Q port amplitude scale multiplier')

        window_duration = vips.getValue('Sampling - duration')

        # Matching can only happen within sampling windows
        for i in range(vips.iterations):
            within_window = False
            for window in vips.sample_windows:
                abs_match_start = utils.get_absolute_time(vips, matching_start, 0, i)
                abs_window_start = utils.get_absolute_time(vips, window['Time'][0], window['Time'][1], i)

                if (abs_match_start >= abs_window_start
                        and abs_match_start + match_duration <= abs_window_start + window_duration):
                    within_window = True
                    break

            if not within_window:
                # Matching was not within any sampling window
                raise ValueError(f'Template matching {m}: The template matching needs to occur within a sampling '
                                 f'window! It is first outside of a sampling window on iteration {i+1}.')

        # Find a pulse to phase sync with, based on the given frequency value
        adjusted_phase = None

        for pulse in vips.pulse_definitions:
            if pulse['Port'] != pulse_port_1:
                continue

            if pulse['Freq'][0] == p_freq:
                # Calculate phase of matching template based on the reference pulse
                reference_time = pulse['Time'][0]
                adjusted_phase = utils.phase_sync(p_freq, p_phase, pulse_start - reference_time)
                break

        # No pulse of matching frequency found
        if adjusted_phase is None:
            raise ValueError(f'Template matching {m}: The given frequency value does not match that of '
                             f'any existing pulse definition!')

        # Construct matching template envelope
        n_points = round(match_duration * vips.sampling_freq)
        envelope1 = templates.get_template_points(vips, envelope1, n_points, 0)
        #envelope2 = templates.get_template_points(vips, envelope2, n_points, 0)
        pad_points = int(padding_length * 4e9)
        envelope1 = np.concatenate((np.zeros(pad_points), envelope1))
        #envelope2 = np.concatenate((np.zeros(pad_points), envelope2))

        # Construct carriers to modulate template with
        end_point = (len(envelope1) - 1) / vips.sampling_freq
        x = np.linspace(0, end_point, len(envelope1))
        # Only fetch the phase shift value if the second port is used
        phase_shift = vips.getValue(f'Port {pulse_port_2} - phase shift') if pulse_port_2 != 0 else 0
        carrier_p1_i = np.cos(2 * np.pi * p_freq * x + np.pi * adjusted_phase)
        carrier_p2_i = np.cos(2 * np.pi * p_freq * x + np.pi * (adjusted_phase + phase_shift + q_relative_phase_offset))
        carrier_p1_q = np.sin(2 * np.pi * p_freq * x + np.pi * adjusted_phase)
        carrier_p2_q = np.sin(2 * np.pi * p_freq * x + np.pi * (adjusted_phase + phase_shift + q_relative_phase_offset))

        # Modulate matching template with carrier
        m_template_i1 = envelope1 * carrier_p1_i * p1_amp_multiplier
        m_template_i2 = envelope1 * carrier_p2_i * p2_amp_multiplier
        m_template_q1 = envelope1 * carrier_p1_q * p1_amp_multiplier
        m_template_q2 = envelope1 * carrier_p2_q * p2_amp_multiplier

        vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_port_1}, '
                          f'template1={m_template_i1}, template2=None)')
        vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_port_1}, '
                          f'template1={m_template_q1}, template2=None)')
        matching_i1 = q.setup_template_matching_pair(sample_port_1, m_template_i1, None)
        matching_q1 = q.setup_template_matching_pair(sample_port_1, m_template_q1, None)
        if sample_port_2 != 0:
            vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_port_2}, '
                              f'template1={m_template_i2}, template2=None)')
            vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_port_2}, '
                              f'template1={m_template_q2}, template2=None)')
            matching_i2 = q.setup_template_matching_pair(sample_port_2, m_template_i2, None)
            matching_q2 = q.setup_template_matching_pair(sample_port_2, m_template_q2, None)

            matchings.append((matching_start, matching_i1, matching_i2, matching_q1, matching_q2, match_duration))
        else:
            matchings.append((matching_start, matching_i1, None, matching_q1, None, match_duration))

    return matchings


def get_port_information(vips, matching_no):
    """
    Get information on what input and output ports are involved in a match,
    and check that everything is set up correctly.
    Return the I and Q input ports (on which the matching is performed) and
    the pulse I and Q ports (where the pulse to be matched is set up).
    """
    sample_i_port = int(vips.getValue(f'Template matching {matching_no} - sampling I port'))
    sample_q_port = vips.getValue(f'Template matching {matching_no} - sampling Q port')
    sample_q_port = utils.combo_to_int(sample_q_port)

    # Matching can only happen on ports with sampling activated
    if sample_i_port not in vips.store_ports or sample_q_port not in [0, *vips.store_ports]:
        raise ValueError(f'Template matching {matching_no}: '
                         f'Sampling needs to be enabled on the ports set as sampling ports!')

    pulse_i_port = int(vips.getValue(f'Template matching {matching_no} - pulse I port'))
    pulse_q_port = vips.getValue(f'Template matching {matching_no} - pulse Q port')
    pulse_q_port = utils.combo_to_int(pulse_q_port)

    # I port must be defined
    if vips.port_settings[pulse_i_port-1]['Mode'] != 'Define':
        raise ValueError(f'Template matching {matching_no}: The pulse I port has to be set to "Define" mode!')

    if pulse_q_port == 0:
        # If the sampling Q port is set to None, the output Q port has to match
        if sample_q_port != 0:
            raise ValueError(f'Template matching {matching_no}: The sampling Q port is defined, but the pulse Q port '
                             f'is set to None. To template match on port {sample_q_port}, ViPS needs '
                             f'to know where the readout pulse is outputted.')

    else:
        # Pulse Q is defined, so sampling Q port also has to be
        if sample_q_port == 0:
            raise ValueError(f'Template matching {matching_no}: The pulse Q port is defined, but the sampling Q port '
                             f'is set to None. If you only wish to template match on one port, there is no need '
                             f'to define two pulse ports.')

    # The port pair needs to consist of different ports
    if pulse_i_port == pulse_q_port:
        raise ValueError(f'Template matching {matching_no}: Pulse I and Q ports cannot have the same port number!')

    if sample_i_port == sample_q_port:
        raise ValueError(f'Template matching {matching_no}: Sampling I and Q ports cannot have the same port number!')

    # If Q is active, it needs to be a copy of I
    if pulse_q_port != 0:
        if (vips.port_settings[pulse_q_port-1]['Mode'] != 'Copy'
                or vips.port_settings[pulse_q_port-1]['Sibling'] != pulse_i_port):
            raise ValueError(f'Template matching {matching_no}: The pulse Q port has to be a copy of the pulse I port!')

    return sample_i_port, sample_q_port, pulse_i_port, pulse_q_port
