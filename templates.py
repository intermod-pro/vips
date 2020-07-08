# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for assembling representations of the
envelope template definitions set up in ViPS.
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d

import input_handling
import envelopes


def get_template_defs(vips, q):
    """
    Get the user-defined templates (consisting of shapes and durations).
    These are represented as dictionaries containing a 'Points' value representing the
    template's shape and a 'Duration' value.
    Long drive templates are a special case, and only contain duration values in the form of a
    'Base' and a 'Delta' value by default. They will also contain 'Rise Points' and 'Fall Points' if
    Gaussian flanks are enabled.
    Return the template definitions in the form of a list.
    """
    num_templates = vips.getValue('Envelope template count')
    template_defs = [{} for _ in range(15)]
    for def_idx in range(1, int(num_templates) + 1):
        template_name = vips.getValue(f'Envelope template {def_idx}: shape')

        # Long drive templates are a special case
        if template_name == 'Long drive':
            template = get_long_drive_definition(vips, def_idx, q.sampling_freq)
        else:
            template = {}
            # Other types share a lot of behaviour
            duration = vips.getValue(f'Envelope template {def_idx}: duration')
            template['Duration'] = duration
            n_points = int(round(duration * q.sampling_freq))
            use_padding = vips.getValue(f'Envelope template {def_idx}: use zero-padding')
            template['Points'] = get_template_points(vips, template_name, n_points, def_idx)

            # Pad with leading zeroes if requested
            if use_padding:
                pad_length = vips.getValue(f'Envelope template {def_idx}: padding length')
                pad_points = int(pad_length * 4)
                template['Points'] = np.concatenate((np.zeros(pad_points), template['Points']))

        template_defs[def_idx - 1] = template
    return template_defs


def get_long_drive_definition(vips, definition_idx, sampling_frequency):
    """
    Construct and return a template definition for a long drive, based on the user-set parameters on
    definition number definition_idx in the instrument.
    """
    template = {}
    dur_string = vips.getValue(f'Envelope template {definition_idx}: long drive duration')
    try:
        template['Base'], template['Delta'] = input_handling.parse_number(dur_string)
    except ValueError:
        error_msg = f'Invalid duration value for template definition {definition_idx}'
        raise ValueError(error_msg)
    # Check if we should add gaussian flanks
    use_gaussian = vips.getValue(f'Envelope template {definition_idx}: use gaussian rise and fall')
    if use_gaussian:
        flank_duration = vips.getValue(f'Envelope template {definition_idx}: gaussian rise and fall duration')
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


def get_template_points(vips, template_name, n_points, definition_idx):
    """
    Return an n_points long list of points forming the shape corresponding to the given template_name.
    definition_idx is needed to fetch extra user-set parameters for certain templates (like the p in sinP).
    """
    if template_name == 'Square':
        return np.ones(n_points+1)[:-1]
    if template_name == 'SinP':
        p = vips.getValue(f'Envelope template {definition_idx}: sinP Value')
        return envelopes.sin_p(p, n_points+1)[:-1]
    if template_name == 'Sin2':
        return envelopes.sin2(n_points+1)[:-1]
    if template_name == 'Sinc':
        cutoff = vips.getValue(f'Envelope template {definition_idx}: sinc cutoff')
        return envelopes.sinc(cutoff, n_points+1)[:-1]
    if template_name == 'Triangle':
        return envelopes.triangle(n_points+1)[:-1]
    if template_name == 'Gaussian':
        trunc = vips.getValue(f'Envelope template {definition_idx}: gaussian truncation')
        return envelopes.gaussian(n_points+1, trunc)[:-1]
    if template_name == 'Cool':
        return envelopes.cool(n_points+1)[:-1]
    if template_name.startswith('Custom'):
        idx = template_name[-1]
        # Fetch the template's shape from the designated input
        custom_template = vips.getValue(f'Custom template {idx}')
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
