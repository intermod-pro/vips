# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for processing various kinds of user input in ViPS.
"""

import re
import math

import numpy as np


def handle_input(quant, value):
    """
    Validate all user input. Depending on the quant's set_cmd value, input might
    be processed in a special way.
    """
    set_commands = quant.set_cmd.replace(' ', '').split(',')

    # Some quants should only accept input of the form [base] + [delta]*i
    if 'time_string' in set_commands:
        # Strip the input down to the essential part
        value = value.replace('INVALID: ', '')
        input_str = value.replace(' ', '')
        # The representation of a number in E notation
        num_rex = r'(([0-9]+)|([0-9]*\.[0-9]+))(e-?[0-9]+)?'
        rex = re.compile(num_rex + '(\\*?i|\\+' + num_rex + '\\*?i)?', re.I)
        # Split the string if the quant accepts multiple values
        if 'list' in set_commands:
            strings = input_str.split(',')
        else:
            strings = [input_str]

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
        return parse_list_of_doubles(value)

    # Some Double quants should only allow integer values
    if 'int' in set_commands:
        return int(value)

    # The quant used for setting padding length allows values in intervals of 0.25
    if 'quarter_value' in set_commands:
        return round(value * 4) / 4

    return value


def is_value_new(vips, quant, value):
    """
    Check if the given value differs from the value stored in the given quant.
    """
    current_value = vips.getValue(quant.name)

    # Combo quants have datatype 2
    if quant.datatype == 2 and isinstance(value, float):
        current_value = vips.getValueIndex(quant.name)

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


def parse_number(string):
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


def parse_list_of_doubles(string):
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