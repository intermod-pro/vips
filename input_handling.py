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

    # Variable names need to be correctly formatted (or empty)
    if 'var_name' in set_commands:
        # Strip the input down to the essential part
        value = value.replace('INVALID:', '')
        value = value.replace(' ', '')
        valid = re.compile('|([A-Za-z_]+[0-9A-Za-z_]*)')
        match = valid.fullmatch(value)
        if not match:
            return f'INVALID: {value}'
        return value

    # Some quants should only accept input of the form [base] + [delta]*i
    if 'time_string' in set_commands:
        is_list = 'list' in set_commands
        return validate_time_string(value, is_list)

    # Some quants should allow a list of comma-separated doubles
    if 'number_string' in set_commands:
        value = value.replace('INVALID: ', '')
        if 'list' not in set_commands:
            try:
                return float(value)
            except ValueError:
                return f'INVALID: {value}'
        else:
            return parse_list_of_doubles(value)

    # Some Double quants should only allow integer values
    if 'int' in set_commands:
        return int(value)

    # The quant used for setting padding length allows values in intervals of 0.25
    if 'quarter_value' in set_commands:
        return round(value * 4) / 4

    return value


def validate_time_string(value, is_list):
    """
    Checks that a "time string" quant is correctly formatted.
    A time string can contain three kinds of expressions, separated by + or -:
        Base values, which are just numeric (with an optional exponent on e form)
        Delta values, which are the same as above with a '*i' or 'i' suffix
        Variables, which are strings that are formatted according to the variable name rules
    If the string is correctly formatted, a polished version is returned.
    Otherwise, the original string is returned with an 'INVALID: ' prefix.
    """
    # Strip the input down to the essential part
    value = value.replace('INVALID:', '')
    input_str = value.replace(' ', '')

    # Split the string if the quant accepts multiple values
    if is_list:
        strings = input_str.split(',')
    else:
        strings = [input_str]

    result = ''
    for idx, s in enumerate(strings):
        accum_string = '' if idx == 0 else ', '
        first = True
        if len(s) == 0:
            return f'INVALID: {value}'
        while len(s) > 0:
            prefix = '[+-]?' if first else '[+-]'
            var_rex = re.compile(prefix + r'[A-Za-z_]+[0-9A-Za-z_]*(\*i)?')
            var_match = var_rex.match(s)
            if var_match:
                # Remove the matched part from input
                match_str = var_match.group()
                match_len = len(match_str)
                s = s[match_len:]
                first = False
                if match_str[0] in ('+', '-'):
                    match_str = f'{match_str[0]} {match_str[1:]}'
                # Add a space after variable name
                accum_string += f'{match_str} '
                continue

            # No variable match, check for numeric value
            num_rex = re.compile(prefix + r'(([0-9]+)|([0-9]*\.[0-9]+))(e-?[0-9]+)?(\*?i)?', re.I)
            num_match = num_rex.match(s)
            if num_match:
                # Remove the matched part from input
                match_str = num_match.group()
                match_len = len(match_str)
                s = s[match_len:]
                first = False

                # Temporarily remove first char if it's a + or -
                if match_str[0] in ('+', '-'):
                    # Put a space after the sign
                    prefix_char = f'{match_str[0]} '
                    match_str = match_str[1:]
                else:
                    prefix_char = ''

                # Perform some cleanup
                while match_str.startswith('0') and len(match_str) > 1 and match_str[1].isnumeric():
                    match_str = match_str[1:]
                # Insert a zero if the number starts with a period
                if match_str.startswith('.'):
                    match_str = '0' + match_str

                match_str = f'{prefix_char}{match_str} '
                match_str = match_str.replace('I', 'i')
                match_str = match_str.replace('e', 'E')
                accum_string += match_str
                continue

            # No match, invalid input
            return f'INVALID: {value}'

        result += accum_string.strip()
    return result


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


def compute_time_string(vips, string):
    """
    Parses a string containing numeric and variable values,
    and returns the summed up base and delta values separately.
    """
    if string.startswith('INVALID:'):
        raise ValueError('Invalid format of time string!')

    string = string.strip()
    parts = string.split(' ')

    latest_sign = 1
    base = 0.0
    delta = 0.0
    for part in parts:
        if part == '+':
            latest_sign = 1
            continue
        if part == '-':
            latest_sign = -1
            continue

        # Literal number
        if part[0].isdigit():
            # Check if this is a delta value
            i_found = re.search('i', part)
            if i_found:
                part = part.replace('*', '')
                part = part.replace('i', '')
                delta += float(part) * latest_sign
                continue

            base += float(part) * latest_sign
            continue

        # Variable
        if part[0].isalpha() or part[0] == '_':
            # Check if this is a delta value
            i_found = re.search(r'\*i', part)
            if i_found:
                part = part.replace('*i', '')
                if part in vips.custom_vars:
                    delta += vips.custom_vars[part] * latest_sign
                    continue
                else:
                    raise ValueError(f'Variable "{part}" is not defined!')

            if part in vips.custom_vars:
                base += vips.custom_vars[part] * latest_sign
                continue
            else:
                raise ValueError(f'Variable "{part}" is not defined!')

        # Failsafe, should never happen
        raise ValueError('Something went wrong when parsing a time string!')

    return base, delta


def parse_list_of_doubles(string):
    """
    Ensure that the given input string is a list of comma-separated floats.
    Return a formatted version of the input string, preceded by 'INVALID: ' if something is incorrect.
    """
    for val in string.split(','):
        try:
            float(val)
        except ValueError:
            return f'INVALID: {string}'
    return string.replace(' ', '').replace(',', ', ').upper()
