# Authored by Johan Blomberg and Gustav Grännsjö, 2020

import ini_generator as generator

gen = generator.Generator()
FILENAME = 'Vivace_Pulse_Sequencer.ini'
N_IN_PORTS = 8
N_OUT_PORTS = 8
MAX_TEMPLATES = 15
CUSTOM_TEMPLATES = 8
MAX_PULSE_DEFS = 16
MAX_MATCHES = 8
MAX_VARIABLES = 10
TEMPLATES = ['Square', 'Long drive', 'Sin2', 'SinP', 'Sinc', 'Triangle', 'Gaussian', 'Cool']
TEMPLATES.extend([f'Custom {i}' for i in range(1, CUSTOM_TEMPLATES + 1)])
MATCH_TEMPLATES = ['Square', 'Sin2', 'Triangle']
MATCH_TEMPLATES.extend([f'Custom {i}' for i in range(1, CUSTOM_TEMPLATES + 1)])
NAME = 'Vivace Pulse Sequencer'
VERSION = '1.3.4'
DRIVER_PATH = 'Vivace_Pulse_Sequencer'
INTERFACE = 'TCPIP'


def section_variables():
    section = 'Custom variables'

    for i in range(1, MAX_VARIABLES+1):
        group = f'Custom variable {i}'

        gen.create_quant(f'Custom variable {i} - name', 'Name', 'STRING', group, section)
        gen.default(f'v{i}')
        gen.tooltip('Variable names can only contain alphanumeric characters and underscores, '
                    'and cannot start with a digit.')
        gen.set_cmd('var_name')

        gen.create_quant(f'Custom variable {i} - value', 'Value', 'DOUBLE', group, section)


def section_general():
    section = 'General settings'
    group = 'Settings'

    # Average
    gen.create_quant('Average', 'Number of averages', 'DOUBLE', group, section)
    gen.limits(low=1)
    gen.default(1)
    gen.set_cmd('int')
    gen.show_in_measurement(True)

    # Trigger period
    gen.create_quant('Trigger period', 'Trigger period', 'DOUBLE', group, section)
    gen.limits(low=0)
    gen.default(200E-6)
    gen.show_in_measurement(True)

    # Iterations
    gen.create_quant('Iterations', 'Iterations', 'DOUBLE', group, section)
    gen.limits(low=1)
    gen.default(1)
    gen.set_cmd('int')
    gen.show_in_measurement(True)

    # -- Output stuff --
    group = 'Time trace output selection'
    # The sampling window to look at
    gen.create_quant('Index of displayed time trace - iteration', 'Iteration', 'DOUBLE', group, section)
    gen.default(1)
    gen.limits(low=1)
    gen.tooltip('Which measurement iteration to output a trace for.')
    gen.set_cmd('int', 'not_affecting_board')
    gen.show_in_measurement(True)

    gen.create_quant('Index of displayed time trace - sample pulse', 'Sampling pulse', 'DOUBLE', group, section)
    gen.default(1)
    gen.limits(low=1)
    gen.tooltip('Which sampling pulse in the given iteration to output a time trace for.')
    gen.set_cmd('int', 'not_affecting_board')
    gen.show_in_measurement(True)


def section_templates():
    section = 'Envelopes'
    group = 'General'

    # Choose how many
    gen.create_quant('Envelope template count', 'No. of envelope templates', 'COMBO', group, section)
    gen.combo_options(*[str(i) for i in range(1, MAX_TEMPLATES+1)])

    # Template definitions
    for i in range(1, MAX_TEMPLATES+1):
        group = f'Envelope template {i}'

        # Template
        gen.create_quant(f'Envelope template {i}: shape', 'Envelope shape', 'COMBO', group, section)
        gen.combo_options(*TEMPLATES)
        gen.visibility(f'Envelope template count', *[str(j) for j in range(i, MAX_TEMPLATES+1)])

        # sinP P
        gen.create_quant(f'Envelope template {i}: sinP Value', 'P', 'DOUBLE', group, section)
        gen.limits(low=0)
        gen.visibility(f'Envelope template {i}: shape', 'SinP')

        # sinc limits
        gen.create_quant(f'Envelope template {i}: sinc cutoff', 'Cutoff', 'DOUBLE', group, section)
        gen.unit('PI')
        gen.default(4)
        gen.limits(low=1e-9)
        gen.visibility(f'Envelope template {i}: shape', 'Sinc')
        gen.tooltip('The sinc wave will be defined from -x*PI to +x*PI')

        # Gaussian truncation
        gen.create_quant(f'Envelope template {i}: gaussian truncation', 'Truncate at x*sigma', 'DOUBLE', group, section)
        gen.default(2)
        gen.limits(low=1E-9)
        gen.visibility(f'Envelope template {i}: shape', 'Gaussian')

        # Duration Double
        gen.create_quant(f'Envelope template {i}: duration', 'Duration', 'DOUBLE', group, section)
        gen.unit('s')
        gen.visibility(f'Envelope template {i}: shape', *[j for j in TEMPLATES if j != 'Long drive'])
        gen.limits(low=1e-9, high=10E-6)
        gen.default(1e-9)

        # Padding settings
        gen.create_quant(f'Envelope template {i}: use zero-padding', 'Use zero-padding', 'BOOLEAN', group, section)
        gen.visibility(f'Envelope template {i}: shape', *[j for j in TEMPLATES if j != 'Long drive'])
        gen.tooltip('Lets you "shift" the template\'s start time by adding up to 2ns of leading zeroes to it.')

        gen.create_quant(f'Envelope template {i}: padding length', 'Padding length', 'DOUBLE', group, section)
        gen.unit('ns')
        gen.visibility(f'Envelope template {i}: use zero-padding', True)
        gen.limits(low=0, high=1.75)
        gen.default(1)
        gen.set_cmd('quarter_value')

        # Long drive special duration
        gen.create_quant(f'Envelope template {i}: long drive duration', 'Duration', 'STRING', group, section)
        gen.unit('s')
        gen.set_cmd('time_string')
        gen.tooltip('Example: 100E6 + 50E6*i')
        gen.default('0 + 0*i')
        gen.visibility(f'Envelope template {i}: shape', 'Long drive')

        # Gaussian rise and fall for long drive bool
        gen.create_quant(f'Envelope template {i}: use gaussian rise and fall', 'Use gaussian rise and fall', 'BOOLEAN', group, section)
        gen.visibility(f'Envelope template {i}: shape', 'Long drive')
        # Gaussian rise/fall duration
        gen.create_quant(f'Envelope template {i}: gaussian rise and fall duration', 'Rise and fall duration', 'DOUBLE', group, section)
        gen.tooltip('Both the rise and the fall have this duration, '
                    'and they are placed within the total duration of the pulse.')
        gen.limits(low=1e-9)
        gen.default(10e-9)
        gen.visibility(f'Envelope template {i}: use gaussian rise and fall', 'True')


def section_port_sequence(port):
    section = f'Port {port} sequence'
    group = 'General'

    # Whether to copy another sequence
    gen.create_quant(f'Port {port} - mode', 'Mode', 'COMBO', group, section)
    gen.combo_options('Disabled', 'Define', 'Copy')

    gen.create_quant(f'Port {port} - copy sequence from', 'Copy from port', 'COMBO', group, section)
    gen.combo_options(*[str(i) for i in range(1, N_OUT_PORTS+1) if i != port])
    gen.visibility(f'Port {port} - mode', 'Copy')

    # If we copy, add the option for phase shifting
    gen.create_quant(f'Port {port} - phase shift', 'Phase shift', 'DOUBLE', group, section)
    gen.unit('PI rad')
    gen.limits(-2, 2)
    gen.visibility(f'Port {port} - mode', 'Copy')

    # If we copy, add the option for amplitude scale scaling
    gen.create_quant(f'Port {port} - amplitude scale multiplier', 'Amplitude scale multiplier', 'DOUBLE', group, section)
    gen.default(1)
    gen.visibility(f'Port {port} - mode', 'Copy')

    # DC shift
    gen.create_quant(f'Port {port} - DC bias', 'DC bias', 'DOUBLE', group, section)
    gen.limits(-0.05, 0.05)
    gen.unit('V')
    gen.tooltip('Range -50mV to +50mV')
    gen.visibility(f'Port {port} - mode', 'Copy', 'Define')

    # Number of groups to display
    gen.small_comment(f'Number of pulses for port {port}')
    gen.create_quant(f'Pulses for port {port}', 'Number of unique pulses', 'COMBO', group, section)
    gen.combo_options(*[str(i) for i in range(1, MAX_PULSE_DEFS+1)])
    gen.visibility(f'Port {port} - mode', 'Define')

    # Definition of individual pulses
    for i in range(1, MAX_PULSE_DEFS+1):
        group = f'Pulse definition {i}'

        # Template
        gen.create_quant(f'Port {port} - def {i} - template', 'Envelope', 'COMBO', group, section)
        gen.combo_options(*[str(j) for j in range(1, MAX_TEMPLATES+1)])
        gen.visibility(f'Pulses for port {port}', *[str(j) for j in range(i, MAX_PULSE_DEFS+1)])

        # Repeat
        gen.create_quant(f'Port {port} - def {i} - repeat count', 'Pulse repeat count', 'DOUBLE', group, section)
        gen.set_cmd('int')
        gen.default(1)
        gen.limits(low=1)
        gen.visibility(f'Pulses for port {port}', *[str(j) for j in range(i, MAX_PULSE_DEFS+1)])

        # Timing
        gen.create_quant(f'Port {port} - def {i} - start times', 'Start times', 'STRING', group, section)
        gen.set_cmd('time_string', 'list')
        gen.unit('s')
        gen.default('0 + 0*i')
        gen.tooltip('Example: 100E6 + 50E6*i, 700E6, ...')
        gen.visibility(f'Pulses for port {port}', *[str(j) for j in range(i, MAX_PULSE_DEFS+1)])

        # Choice of sine generator
        gen.create_quant(f'Port {port} - def {i} - sine generator', 'Sine generator (DRAG)', 'COMBO', group, section)
        gen.combo_options('1', '2', 'DRAG', 'None')
        gen.visibility(f'Pulses for port {port}', *[str(j) for j in range(i, MAX_PULSE_DEFS+1)])
        gen.tooltip('Choose which sine generator to output this pulse on, or configure DRAG over both. '
                    'If set to None, the Frequency and Phase parameters will have no effect on this pulse.')

        # DRAG parameters
        gen.create_quant(f'Port {port} - def {i} - DRAG sibling port', 'DRAG sibling port', 'COMBO', group, section)
        gen.combo_options(*[str(i) for i in range(1, N_OUT_PORTS+1) if i != port])
        gen.visibility(f'Port {port} - def {i} - sine generator', 'DRAG')

        gen.create_quant(f'Port {port} - def {i} - DRAG phase shift', 'Sibling phase shift', 'DOUBLE', group, section)
        gen.unit('PI rad')
        gen.limits(-2, 2)
        gen.visibility(f'Port {port} - def {i} - sine generator', 'DRAG')

        gen.create_quant(f'Port {port} - def {i} - DRAG scale', 'DRAG scale', 'DOUBLE', group, section)
        gen.default(1e-9)
        gen.unit('s')
        gen.visibility(f'Port {port} - def {i} - sine generator', 'DRAG')

        gen.create_quant(f'Port {port} - def {i} - DRAG detuning frequency', 'DRAG detuning frequency', 'DOUBLE', group, section)
        gen.unit('Hz')
        gen.limits(low=0)
        gen.visibility(f'Port {port} - def {i} - sine generator', 'DRAG')

        # Sweep param
        gen.create_quant(f'Port {port} - def {i} - Sweep param', 'Sweepable parameter', 'COMBO', group, section)
        gen.combo_options('None', 'Amplitude scale', 'Carrier frequency', 'Phase')
        gen.visibility(f'Port {port} - def {i} - sine generator', '1', '2', 'DRAG', 'None')

        # Params: amp
        gen.create_quant(f'Port {port} - def {i} - amp', 'Amplitude scale', 'DOUBLE', group, section)
        gen.limits(-1, 1)
        gen.default(1)
        gen.visibility(f'Port {port} - def {i} - Sweep param', 'None', 'Carrier frequency', 'Phase')

        # Params: freq
        gen.create_quant(f'Port {port} - def {i} - freq', 'Carrier frequency', 'DOUBLE', group, section)
        gen.unit('Hz')
        gen.limits(0, 2E9)
        gen.visibility(f'Port {port} - def {i} - Sweep param', 'None', 'Amplitude scale', 'Phase')

        # Params: phase
        gen.create_quant(f'Port {port} - def {i} - phase', 'Phase', 'DOUBLE', group, section)
        gen.unit('PI rad')
        gen.limits(-2, 2)
        gen.visibility(f'Port {port} - def {i} - Sweep param', 'None', 'Amplitude scale', 'Carrier frequency')

        # How?
        gen.create_quant(f'Port {port} - def {i} - Sweep format', 'Sweep format', 'COMBO', group, section)
        gen.combo_options('Linear: Start-End', 'Linear: Center-Span', 'Custom')
        gen.visibility(f'Port {port} - def {i} - Sweep param', 'Amplitude scale', 'Carrier frequency', 'Phase')

        # Linear: Start-End
        gen.create_quant(f'Port {port} - def {i} - Sweep linear start', 'Start', 'DOUBLE', group, section)
        gen.visibility(f'Port {port} - def {i} - Sweep format', 'Linear: Start-End')
        gen.create_quant(f'Port {port} - def {i} - Sweep linear end', 'End', 'DOUBLE', group, section)
        gen.visibility(f'Port {port} - def {i} - Sweep format', 'Linear: Start-End')

        # Linear: Center-Span
        gen.create_quant(f'Port {port} - def {i} - Sweep linear center', 'Center', 'DOUBLE', group, section)
        gen.visibility(f'Port {port} - def {i} - Sweep format', 'Linear: Center-Span')
        gen.create_quant(f'Port {port} - def {i} - Sweep linear span', 'Span', 'DOUBLE', group, section)
        gen.visibility(f'Port {port} - def {i} - Sweep format', 'Linear: Center-Span')

        # Custom steps
        gen.create_quant(f'Port {port} - def {i} - Sweep custom steps', 'Step values', 'STRING', group, section)
        gen.tooltip('Separate with comma')
        gen.set_cmd('number_string', 'list')
        gen.visibility(f'Port {port} - def {i} - Sweep format', 'Custom')

        # Conditionals
        gen.create_quant(f'Port {port} - def {i} - Template matching condition 1',
                         'Matching condition 1', 'COMBO', group, section)
        gen.combo_options('None', *[str(j) for j in range(1, MAX_MATCHES+1)])
        gen.tooltip('This pulse will only be outputted if the selected template matching yields a positive result.')
        gen.visibility(f'Pulses for port {port}', *[str(j) for j in range(i, MAX_PULSE_DEFS + 1)])

        gen.create_quant(f'Port {port} - def {i} - Template matching condition 1 quadrature',
                         'Condition 1 - quadrature', 'COMBO', group, section)
        gen.combo_options('I', 'Q')
        gen.visibility(f'Port {port} - def {i} - Template matching condition 1', *[str(j) for j in range(1, MAX_MATCHES+1)])

        gen.create_quant(f'Port {port} - def {i} - Template matching condition 2',
                         'Matching condition 2', 'COMBO', group, section)
        gen.combo_options('None', *[str(j) for j in range(1, MAX_MATCHES + 1)])
        gen.tooltip('This pulse will only be outputted if both of the selected template matches yield a positive result.')
        gen.visibility(f'Port {port} - def {i} - Template matching condition 1', *[str(j) for j in range(1, MAX_PULSE_DEFS + 1)])

        gen.create_quant(f'Port {port} - def {i} - Template matching condition 2 quadrature',
                         'Condition 2 - quadrature', 'COMBO', group, section)
        gen.combo_options('I', 'Q')
        gen.visibility(f'Port {port} - def {i} - Template matching condition 2',
                       *[str(j) for j in range(1, MAX_MATCHES + 1)])


def section_sample():
    section = 'Sampling'
    group = 'Timing'

    # Timings
    gen.create_quant(f'Sampling - start times', 'Start times', 'STRING', group, section)
    gen.set_cmd('time_string', 'list')
    gen.unit('s')
    gen.default('0 + 0*i')
    gen.tooltip('Example: 100E6 + 50E6*i, 700E6, ...')

    # Duration
    gen.create_quant('Sampling - duration', 'Duration', 'DOUBLE', group, section)
    gen.limits(0, 4096E-9)
    gen.unit('s')

    group = 'Port selection'
    # Port selection
    for i in range(1, N_IN_PORTS+1):
        gen.create_quant(f'Sampling on port {i}', f'Port {i}', 'BOOLEAN', group, section)

    # Result values (readonly vectors, so they aren't visible)
    for i in range(1, N_IN_PORTS+1):
        gen.create_quant(f'Port {i}: Time trace', '', 'VECTOR', group, section)
        gen.get_cmd('get_trace')
        gen.add_line('x_unit: s')
        gen.visibility(f'Sampling on port {i}', True)
        gen.show_in_measurement(True)

    # Custom template input vectors
    for i in range(1, CUSTOM_TEMPLATES + 1):
        gen.create_quant(f'Custom template {i}', '', 'VECTOR', group, section)
        gen.permission('WRITE')

    # Template previews
    for template in range(1, MAX_TEMPLATES+1):
        gen.create_quant(f'Template {template}: Preview', '', 'VECTOR', group, section)
        gen.get_cmd('template_preview')
        gen.add_line('x_unit: s')
        gen.visibility(f'Envelope template {template}: shape', *TEMPLATES)


def section_matching():
    section = 'Template matching'
    group = 'General'

    gen.create_quant('Number of matches', 'Number of matches', 'COMBO', group, section)
    gen.combo_options(*range(MAX_MATCHES+1))

    for m in range(1, MAX_MATCHES+1):
        group = f'Matching {m}'

        gen.create_quant(f'Template matching {m} - template 1', 'Template 1', 'COMBO', group, section)
        gen.combo_options(*MATCH_TEMPLATES)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('The template to match against. '
                    'The envelope given will be modulated with the frequency you specify further below.')
        # TODO
        #gen.create_quant(f'Template matching {m} - template 2', 'Template 2', 'COMBO', group, section)
        #gen.combo_options('Zeroes', *MATCH_TEMPLATES)
        #gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        #gen.tooltip('This optional template will be used for comparison with the first template.')

        gen.create_quant(f'Template matching {m} - sampling I port', 'Sampling I port', 'COMBO', group, section)
        gen.combo_options(*range(1, N_IN_PORTS + 1))
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.create_quant(f'Template matching {m} - sampling Q port', 'Sampling Q port', 'COMBO', group, section)
        gen.combo_options('None', *range(1, N_IN_PORTS + 1))
        gen.default('2')
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])

        gen.create_quant(f'Template matching {m} - matching start time', 'Matching start time', 'DOUBLE', group, section)
        gen.limits(low=0)
        gen.unit('s')
        gen.set_cmd('quarter_nanos')
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES+1)])
        gen.tooltip('The time at which matching should start.')

        gen.create_quant(f'Template matching {m} - matching duration', 'Matching duration', 'DOUBLE', group, section)
        gen.limits(0, 1020e-9)
        gen.unit('s')
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('How long matching should last. '
                    'Should ideally be equal to the duration of the pulse you match with.')

        gen.create_quant(f'Template matching {m} - pulse I port', 'Pulse I port', 'COMBO', group, section)
        gen.combo_options(*range(1, N_IN_PORTS + 1))
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.create_quant(f'Template matching {m} - pulse Q port', 'Pulse Q port', 'COMBO', group, section)
        gen.combo_options('None', *range(1, N_IN_PORTS + 1))
        gen.default('2')
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])

        gen.create_quant(f'Template matching {m} - pulse start time', 'Pulse start time', 'DOUBLE', group, section)
        gen.limits(low=0)
        gen.unit('s')
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('The start time of the pulse you match with. '
                    'Used to calculate phase sync for the matching template.')

        gen.create_quant(f'Template matching {m} - pulse frequency', 'Pulse frequency', 'DOUBLE', group, section)
        gen.unit('Hz')
        gen.limits(0, 2E9)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('Frequency to modulate the matching template with. '
                    'Should ideally be equal to the frequency of the pulse you match with.')

        gen.create_quant(f'Template matching {m} - pulse phase', 'Pulse phase', 'DOUBLE', group, section)
        gen.unit('PI rad')
        gen.limits(-2, 2)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('Phase offset to apply to the matching template. '
                    'Should be equal to the phase offset of the pulse you match with.')

        gen.create_quant(f'Template matching {m} - Q port phase adjustment', 'Q port phase adjustment',
                         'DOUBLE', group, section)
        gen.unit('PI rad')
        gen.limits(-2, 2)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('This value will be added to the Q port\'s phase to account for phase shifts caused by the mixers.')

        gen.create_quant(f'Template matching {m} - I port amplitude scale multiplier', 'I port amplitude scale multiplier',
                         'DOUBLE', group, section)
        gen.default(1)
        gen.limits(0, 1)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('This value will be used to scale the matching template to the amplitude of the I port\'s output.')
        gen.create_quant(f'Template matching {m} - Q port amplitude scale multiplier',
                         'Q port amplitude scale multiplier',
                         'DOUBLE', group, section)
        gen.default(1)
        gen.limits(0, 1)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.tooltip('This value will be used to scale the matching template to the amplitude of the Q port\'s output.')

        gen.create_quant(f'Template matching {m}: Results', '', 'VECTOR_COMPLEX', group, section)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.get_cmd('get_match')
        gen.permission('READ')

    for m in range(1, MAX_MATCHES+1):
        gen.create_quant(f'Template matching {m}: Average result', f'Average {m}', 'COMPLEX',
                         'Averages (read-only)', section)
        gen.visibility('Number of matches', *[str(i) for i in range(m, MAX_MATCHES + 1)])
        gen.permission('READ')
        gen.get_cmd('get_match')


def section_preview():
    section = 'Preview'
    group = 'Settings'

    gen.create_quant('Preview port', 'Preview sequence on port', 'COMBO', group, section)
    gen.combo_options(*[i for i in range(1, N_OUT_PORTS+1)])
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Preview iteration', 'Preview iteration', 'DOUBLE', group, section)
    gen.default(1)
    gen.limits(low=1)
    gen.set_cmd('int', 'not_affecting_board')

    gen.create_quant('Enable preview slicing', 'Enable preview slicing', 'BOOLEAN', group, section)
    gen.tooltip('This will let you specify which segment of the pulse sequence to preview.')
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Preview slice start', 'Slice start', 'DOUBLE', group, section)
    gen.unit('s')
    gen.limits(low=0)
    gen.visibility('Enable preview slicing', True)
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Preview slice end', 'Slice end', 'DOUBLE', group, section)
    gen.unit('s')
    gen.limits(low=1E-9)
    gen.default(1E-9)
    gen.visibility('Enable preview slicing', True)
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Preview sample windows', 'Preview sample windows', 'BOOLEAN', group, section)
    gen.tooltip('This will display sample windows as flat lines at y=-0.1. '
                'Output pulses that overlap with these sample windows will be hidden.')
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Pulse sequence preview', '', 'VECTOR', group, section)
    gen.get_cmd('sequence_preview')
    gen.permission('READ')


def section_debug():
    section = 'Debug'
    group = 'Logging'

    gen.create_quant('Enable Vivace call logging', 'Enable Vivace call logging', 'BOOLEAN', group, section)
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Log file name', 'Log file name', 'STRING', group, section)
    gen.visibility('Enable Vivace call logging', True)
    gen.default('log')
    gen.set_cmd('not_affecting_board')

    gen.create_quant('Overwrite previous log', 'Overwrite previous log', 'BOOLEAN', group, section)
    gen.visibility('Enable Vivace call logging', True)
    gen.default(True)
    gen.set_cmd('not_affecting_board')

    group = 'Dry run'

    gen.create_quant('Vivace connection enabled', 'Enable connection to Vivace', 'BOOLEAN', group, section)
    gen.default(True)
    gen.set_cmd('vivace_connect')
    gen.tooltip('When disabled, ViPS will not send any data to the Vivace hardware. '
                'All output will default to dummy values.')

    group = 'Versions'

    gen.create_quant('ViPS version', 'ViPS', 'STRING', group, section)
    gen.set_cmd('vips_version')

    gen.create_quant('Vivace firmware version', 'Vivace firmware', 'STRING', group, section)
    gen.set_cmd('vivace_fw_version')

    gen.create_quant('Vivace server version', 'Vivace server', 'STRING', group, section)
    gen.set_cmd('vivace_server_version')

    gen.create_quant('Vivace API version', 'Vivace API', 'STRING', group, section)
    gen.set_cmd('vivace_api_version')


########## INIT ##########
gen.general_settings(NAME, VERSION, DRIVER_PATH, author='Johan Blomberg and Gustav Grännsjö', interface=INTERFACE)

gen.big_comment('CUSTOM VARIABLES')
section_variables()

# TEMPLATES
gen.big_comment('TEMPLATES')
section_templates()

gen.big_comment('SWEEPABLE PULSE (and more...)')
section_general()

for p in range(1, N_OUT_PORTS+1):
    gen.big_comment(f'Pulse definitions - Port {p}')
    section_port_sequence(p)

gen.big_comment('SAMPLING')
section_sample()

gen.big_comment('TEMPLATE MATCHING')
section_matching()

gen.big_comment('PREVIEW')
section_preview()

section_debug()

gen.write(FILENAME)
