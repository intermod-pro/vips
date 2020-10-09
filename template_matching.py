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
        (sample_port_1, use_port_pair) = get_port_information(vips, m)

        envelope = vips.getValue(f'Template matching {m} - template')

        matching_start = vips.getValue(f'Template matching {m} - matching start time')
        padding_length = ((matching_start * 1e9) % 2) / 1e9
        matching_start = matching_start - padding_length
        match_duration = vips.getValue(f'Template matching {m} - matching duration')

        freq = vips.getValue(f'Template matching {m} - frequency')
        phase = vips.getValue(f'Template matching {m} - template phase')

        p2_phase_shift = vips.getValue(f'Template matching {m} - second port phase shift')
        p1_amp_multiplier = vips.getValue(f'Template matching {m} - first port amplitude scale multiplier')
        p2_amp_multiplier = vips.getValue(f'Template matching {m} - second port amplitude scale multiplier')

        threshold = vips.getValue(f'Template matching {m} - threshold')

        window_duration = vips.sampling_duration

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

        # Construct matching template envelope
        n_points = round(match_duration * vips.sampling_freq)
        envelope = templates.get_template_points(vips, envelope, n_points, 0)
        pad_points = int(padding_length * 4e9)
        envelope = np.concatenate((np.zeros(pad_points), envelope))

        # Construct carriers to modulate template with
        end_point = (len(envelope) - 1) / vips.sampling_freq
        x = np.linspace(0, end_point, len(envelope))

        # If we use two ports, set up the templates for IQ combination
        if use_port_pair:
            carrier_m1_p1 = np.cos(2 * np.pi * freq * x + np.pi * phase)
            carrier_m1_p2 = np.sin(2 * np.pi * freq * x + np.pi * (phase + p2_phase_shift))
            carrier_m2_p1 = -np.sin(2 * np.pi * freq * x + np.pi * phase)
            carrier_m2_p2 = np.cos(2 * np.pi * freq * x + np.pi * (phase + p2_phase_shift))
        else:
            carrier_m1_p1 = np.cos(2 * np.pi * freq * x + np.pi * phase)
            carrier_m1_p2 = 0
            carrier_m2_p1 = np.sin(2 * np.pi * freq * x + np.pi * phase)
            carrier_m2_p2 = 0

        # Modulate matching template with carrier
        template_m1_p1 = envelope * carrier_m1_p1 * p1_amp_multiplier
        template_m1_p2 = envelope * carrier_m1_p2 * p2_amp_multiplier
        template_m2_p1 = envelope * carrier_m2_p1 * p1_amp_multiplier
        template_m2_p2 = envelope * carrier_m2_p2 * p2_amp_multiplier

        vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_port_1}, template1={template_m1_p1}, '
                          f'template2={template_m1_p2}, threshold={threshold}, port_pair={use_port_pair}')
        vips.lgr.add_line(f'q.setup_template_matching_pair(port={sample_port_1}, template1={template_m2_p1}, '
                          f'template2={template_m2_p2}, threshold={threshold}, port_pair={use_port_pair}')
        matching_i = q.setup_template_matching_pair(sample_port_1, template_m1_p1, template_m1_p2, threshold / match_duration, use_port_pair)
        matching_q = q.setup_template_matching_pair(sample_port_1, template_m2_p1, template_m2_p2, threshold / match_duration, use_port_pair)

        matchings.append((matching_start, match_duration, matching_i, matching_q))

    return matchings


def get_port_information(vips, matching_no):
    """
    Get information on what sampling ports are used in a match, and check that
    everything is set up correctly.
    Return the number of the first port, and a boolean indicating if the second port is used.
    """
    sample_port_1 = int(vips.getValue(f'Template matching {matching_no} - first sampling port'))
    use_port_2 = vips.getValue(f'Template matching {matching_no} - match on two ports')
    sample_port_2 = sample_port_1 + 1 if use_port_2 else 0

    if sample_port_2 > 8:
        raise ValueError(f'Cannot perform template matching on two ports if the first is port 8!')

    # Matching can only happen on ports with sampling activated
    if sample_port_1 not in vips.sampling_ports or sample_port_2 not in [0, *vips.sampling_ports]:
        raise ValueError(f'Template matching {matching_no}: '
                         f'Sampling needs to be enabled on the ports set as sampling ports!')

    return sample_port_1, use_port_2
