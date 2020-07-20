# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of various utility functions used by different parts of the ViPS driver.
"""

import math
import numpy as np


def template_def_to_points(vips, template_def, iteration):
    """
    Calculate X and Y point values for the given template definition and iteration.
    Return X, Y.
    """
    if 'Rise Points' in template_def:  # Gauss long drive
        duration = template_def['Base'] + template_def['Delta'] * iteration
        rise_p = template_def['Rise Points']
        n_long_points = round((duration - 2 * template_def['Flank Duration']) * vips.sampling_freq)
        if n_long_points >= 1e5:
            raise ValueError('Your long drive template is too long to preview, '
                             'it would take forever to create a list of >100 000 points!')
        long_p = np.linspace(1, 1, n_long_points)
        fall_p = template_def['Fall Points']
        total_points = len(rise_p) + n_long_points + len(fall_p)
        end_point = (total_points - 1) / vips.sampling_freq
        x = np.linspace(0, end_point, total_points)
        y = np.concatenate((rise_p, long_p, fall_p))
        return x, y

    if 'Base' in template_def:  # Long drive
        duration = template_def['Base'] + template_def['Delta'] * iteration
        if duration <= 0:
            return [], []
        n_points = round(duration * vips.sampling_freq)
        if n_points >= 1e5:
            raise ValueError('Your long drive template is too long to preview, '
                             'it would take forever to create a list of >100 000 points!')
        end_point = (n_points - 1) / vips.sampling_freq
        x = np.linspace(0, end_point, n_points)
        y = np.linspace(1, 1, n_points)
        return x, y

    if template_def['Duration'] <= 0:
        return [], []
    y = template_def['Points']
    end_point = (len(y) - 1) / vips.sampling_freq
    x = np.linspace(0, end_point, len(y))
    return x, y


def phase_sync(frequency, phase, elapsed_time):
    """
    Calculate what the phase should be for a carrier at the given frequency
    after the given time has elapsed.
    """
    return phase + (elapsed_time * frequency * 2)


def get_amp_freq_phase(pulse, iteration):
    """
    Return the amplitude, frequency and phase parameters for a given
    pulse definition in the given iteration step.
    """
    p_amp = pulse['Amp'][iteration]
    p_freq = pulse['Freq'][iteration]
    p_phase = pulse['Phase'][iteration]
    return p_amp, p_freq, p_phase


def get_absolute_time(vips, base, delta, iteration):
    """
    Given a base, a delta and the current iteration index, compute and return an
    absolute time value for use with Vivace's methods.
    """
    return (vips.measurement_period * iteration  # The current period's start time
            + base  # The constant part of the given time
            + delta * iteration)  # The scaling part of the given time


def are_fp_pairs_close(tuptup1, tuptup2):
    """
    Compare two objects to see if their numerical values are identical within
    a small margin of error. The objects each contain a pair of frequency-phase
    tuples, so they each have 4 values that need to be compared against the
    other object.
    """
    return (math.isclose(tuptup1[0][0], tuptup2[0][0]) and
            math.isclose(tuptup1[0][1], tuptup2[0][1]) and
            math.isclose(tuptup1[1][0], tuptup2[1][0]) and
            math.isclose(tuptup1[1][1], tuptup2[1][1]))
