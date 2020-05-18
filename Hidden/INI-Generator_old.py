class Generator:

    FILENAME = 'FPGA_Test.INI'
    contents = ''
    boilerplate = """[General settings]

# The name is shown in all the configuration windows
name: FPGA Test

# The version string
version: 0.1

# Name of folder containing the code defining a custom driver
driver_path: FPGA_Test

[VISA settings]

# Enable or disable communication over the VISA protocol (True or False)
# If False, the driver will not perform any operations (unless there is a custom driver).
use_visa: False


"""

    def __init__(self):
        # Run the thing
        with open(self.FILENAME, 'w') as f:
            # boilerplate
            self.contents += self.boilerplate

            self.big_comment("Outputs")
            self.section_output()

            self.big_comment("Pulses")
            self.section_pulses()

            self.big_comment("Templates")
            self.section_templates()

            self.big_comment("Structure")
            self.section_structure()

            f.write(self.contents)

    def big_comment(self, string):
        self.contents += """###############################
### {}
###############################

""".format(string + ' ' + '#'*(26-len(string)))

    def small_comment(self, string):
        self.contents += '### {}\n'.format(string + ' ' + '#'*(26-len(string)))

    def create_quant(self, name, label, type, group, section):
        self.contents += f"""
[{name}]
label: {label}
datatype: {type}
group: {group}
section: {section}

"""

    def combo_options(self, options):
        str = ''
        for i, o in enumerate(options):
            str += f'combo_def_{i+1}: {o}\n'

        self.contents += str

    # For quants with visibility controlled by another quant
    def visibility(self, quant, values):
        str = f'state_quant: {quant}\n'

        for i, v in enumerate(values):
            str += f'state_value_{i+1}: {v}\n'

        self.contents += str

    # When we only want to add a single line
    def add_line(self, string):
        self.contents += string + '\n'

    def section_output(self):
        section = 'Output ports'
        for p in range(1, 9):
            self.small_comment(f'Port number {p}')
            group = f'Port {p}'
            enable_box = f'Enable output, port {p}'

            # Enable-box
            self.create_quant(enable_box, 'Enabled', 'BOOLEAN', group, section)
            self.add_line('def_value: False')

            group = f'Port {p} - Frequency and phase'

            # Rows in LUT
            self.create_quant(f'Freq & phase points, port {p}', 'No. of Freq. and Phase Values', 'DOUBLE', group, section)
            # Visibility
            self.visibility(enable_box, ['True'])
            self.add_line('def_value: 1')
            self.add_line('')

            # ----------------- Freq type ----------------
            self.create_quant(f'Freq value format, port {p}', 'Frequency value format', 'COMBO', group, section)
            self.combo_options(['Constant', 'Linear: Start/End', 'Linear: Center/Span', 'Custom'])
            self.visibility(enable_box, ['True'])

            self.small_comment(f'Port {p} - Frequency - Constant')
            # Constant
            self.create_quant(f'Freq Constant, port {p}', 'Value', 'DOUBLE', group, section)
            self.visibility(f'Freq value format, port {p}', ['Constant'])
            self.add_line('unit: Hz')
            self.add_line('low_lim: 1')
            self.add_line('high_lim: 2E9')

            self.small_comment(f'Port {p} - Frequency - Linear: Start/End')
            # Linear: Start/End
            self.create_quant(f'Freq Linear Start, port {p}', 'Start', 'DOUBLE', group, section)
            self.visibility(f'Freq value format, port {p}', ['Linear: Start/End'])
            self.add_line('unit: Hz')
            self.add_line('low_lim: 1')
            self.add_line('high_lim: 2E9')
            self.create_quant(f'Freq Linear End, port {p}', 'End', 'DOUBLE', group, section)
            self.visibility(f'Freq value format, port {p}', ['Linear: Start/End'])
            self.add_line('unit: Hz')
            self.add_line('low_lim: 1')
            self.add_line('high_lim: 2E9')

            self.small_comment(f'Port {p} - Frequency - Linear: Center/Span')
            # Linear: Center/Span
            self.create_quant(f'Freq Linear Center, port {p}', 'Center', 'DOUBLE', group, section)
            self.visibility(f'Freq value format, port {p}', ['Linear: Center/Span'])
            self.add_line('unit: Hz')
            self.add_line('low_lim: 1')
            self.add_line('high_lim: 2E9')
            self.create_quant(f'Freq Linear Span, port {p}', 'Span', 'DOUBLE', group, section)
            self.visibility(f'Freq value format, port {p}', ['Linear: Center/Span'])
            self.add_line('unit: Hz')
            self.add_line('low_lim: 0')

            self.small_comment(f'Port {p} - Frequency - Custom')
            # Custom frequency LUT values
            self.create_quant(f'Custom Frequency length, port {p}', 'Length of custom sequence', 'COMBO', group, section)
            self.add_line('tooltip: This sequence will be repeated until its length is the number of values specified above.')
            self.visibility(f'Freq value format, port {p}', ['Custom'])
            self.combo_options([str(i) for i in range(1, 11)])

            for i in range(1, 11):
                self.create_quant(f'Custom Frequency no. {i}, port {p}', f'Frequency {i}', 'DOUBLE', group, section)
                self.add_line('unit: Hz')
                self.add_line('low_lim: 0')
                self.add_line('high_lim: 2E9')
                self.visibility(f'Custom Frequency length, port {p}', [str(j) for j in range(i, 11)])


            # ----------------- Phase type ----------------
            self.create_quant(f'Phase value format, port {p}', 'Phase value format', 'COMBO', group, section)
            self.combo_options(['Constant', 'Linear: Start/End', 'Linear: Center/Span', 'Custom'])
            self.visibility(enable_box, ['True'])

            self.small_comment(f'Port {p} - Phase - Constant')
            # Constant
            self.create_quant(f'Phase Constant, port {p}', 'Value', 'DOUBLE', group, section)
            self.visibility(f'Phase value format, port {p}', ['Constant'])
            self.add_line('unit: PI rad')
            self.add_line('low_lim: -2')
            self.add_line('high_lim: 2')


            self.small_comment(f'Port {p} - Phase - Linear: Start/End')
            #Linear: Start/End
            self.create_quant(f'Phase Linear Start, port {p}', 'Start', 'DOUBLE', group, section)
            self.visibility(f'Phase value format, port {p}', ['Linear: Start/End'])
            self.add_line('unit: PI rad')
            self.add_line('low_lim: -2')
            self.add_line('high_lim: 2')
            self.add_line('def_value: 0')
            self.create_quant(f'Phase Linear End, port {p}', 'End', 'DOUBLE', group, section)
            self.visibility(f'Phase value format, port {p}', ['Linear: Start/End'])
            self.add_line('unit: Hz')
            self.add_line('low_lim: -2')
            self.add_line('high_lim: 2')
            self.add_line('def_value: 0')

            self.small_comment(f'Port {p} - Phase - Linear: Center/Span')
            # Linear: Center/Span
            self.create_quant(f'Phase Linear Center, port {p}', 'Center', 'DOUBLE', group, section)
            self.visibility(f'Phase value format, port {p}', ['Linear: Center/Span'])
            self.add_line('unit: PI rad')
            self.add_line('low_lim: -2')
            self.add_line('high_lim: 2')
            self.add_line('def_value: 0')

            self.create_quant(f'Phase Linear Span, port {p}', 'Span', 'DOUBLE', group, section)
            self.visibility(f'Phase value format, port {p}', ['Linear: Center/Span'])
            self.add_line('unit: PI rad')
            self.add_line('low_lim: 0')
            self.add_line('def_value: 0')

            self.small_comment(f'Port {p} - Phase - Custom')
            # Custom phase LUT values
            self.create_quant(f'Custom phase length, port {p}', 'Length of custom sequence', 'COMBO', group,
                              section)
            self.add_line('tooltip: This sequence will be repeated until its length is the number of values specified above.')
            self.visibility(f'Phase value format, port {p}', ['Custom'])
            self.combo_options([str(i) for i in range(1, 11)])
            for i in range(1, 11):
                self.create_quant(f'Custom Phase no. {i}, port {p}', f'Phase {i}', 'DOUBLE', group, section)
                self.visibility(f'Custom phase length, port {p}', [str(j) for j in range(i, 11)])
                self.add_line('unit: PI rad')
                self.add_line('low_lim: -2')
                self.add_line('high_lim: 2')

            #----------------- AMPLITUDE --------------
            group = f'Port {p} - Amplitude scaling'
            # Rows in LUT
            self.create_quant(f'Amplitude points, port {p}', 'No. of amplitude scale values', 'DOUBLE', group,
                              section)
            # Visibility
            self.visibility(enable_box, ['True'])
            self.add_line('def_value: 1')
            self.add_line('')

            # ----------------- Amp type ----------------
            self.create_quant(f'Amplitude value format, port {p}', 'Amplitude value format', 'COMBO', group, section)
            self.combo_options(['Constant', 'Linear: Start/End', 'Linear: Center/Span', 'Custom'])
            self.visibility(enable_box, ['True'])

            self.small_comment(f'Port {p} - Amplitude - Constant')
            # Constant
            self.create_quant(f'Amplitude Constant, port {p}', 'Value', 'DOUBLE', group, section)
            self.visibility(f'Amplitude value format, port {p}', ['Constant'])
            self.add_line('unit: FS')
            self.add_line('low_lim: -1')
            self.add_line('high_lim: 1')

            self.small_comment(f'Port {p} - Amplitude - Linear: Start/End')
            # Linear: Start/End
            self.create_quant(f'Amplitude Linear Start, port {p}', 'Start', 'DOUBLE', group, section)
            self.visibility(f'Amplitude value format, port {p}', ['Linear: Start/End'])
            self.add_line('unit: FS')
            self.add_line('low_lim: -1')
            self.add_line('high_lim: 1')
            self.create_quant(f'Amplitude Linear End, port {p}', 'End', 'DOUBLE', group, section)
            self.visibility(f'Amplitude value format, port {p}', ['Linear: Start/End'])
            self.add_line('unit: FS')
            self.add_line('low_lim: -1')
            self.add_line('high_lim: 1')

            self.small_comment(f'Port {p} - Amplitude - Linear: Center/Span')
            # Linear: Center/Span
            self.create_quant(f'Amplitude Linear Center, port {p}', 'Center', 'DOUBLE', group, section)
            self.visibility(f'Amplitude value format, port {p}', ['Linear: Center/Span'])
            self.add_line('unit: FS')
            self.add_line('low_lim: -1')
            self.add_line('high_lim: 1')
            self.create_quant(f'Amplitude Linear Span, port {p}', 'Span', 'DOUBLE', group, section)
            self.visibility(f'Amplitude value format, port {p}', ['Linear: Center/Span'])
            self.add_line('unit: FS')
            self.add_line('low_lim: 0')

            self.small_comment(f'Port {p} - Amplitude - Custom')
            # Custom frequency LUT values
            self.create_quant(f'Custom Amplitude length, port {p}', 'Length of custom sequence', 'COMBO', group,
                              section)
            self.add_line(
                'tooltip: This sequence will be repeated until its length is the number of values specified above.')
            self.visibility(f'Amplitude value format, port {p}', ['Custom'])
            self.combo_options([str(i) for i in range(1, 11)])

            for i in range(1, 11):
                self.create_quant(f'Custom Amplitude no. {i}, port {p}', f'Amplitude scale {i}', 'DOUBLE', group, section)
                self.add_line('unit: FS')
                self.add_line('low_lim: -1')
                self.add_line('high_lim: 1')
                self.visibility(f'Custom Amplitude length, port {p}', [str(j) for j in range(i, 11)])

    def section_pulses(self):
        section = 'Pulse definitions'
        group = 'General'

        self.small_comment("General pulse settings")


        # 'hidden' box
        self.create_quant('Pulse amount', 'Number of defined pulses', 'COMBO', group, section)
        self.combo_options([str(i) for i in range(1, 129)])

        self.small_comment("PULSES")

        for i in range(1, 129):
            group = f'Pulse {i}'

            # Single/double pulse
            self.create_quant(f'Pulse {i} port single/double', 'Single/double port', 'COMBO', group, section)
            self.combo_options(['Double', 'Single'])
            self.visibility('Pulse amount', [str(j) for j in range(i, 129)])

            # Port I
            self.create_quant(f'Pulse {i} port I', 'Port I', 'COMBO', group, section)
            self.combo_options([str(j) for j in range(1, 9)])
            self.visibility(f'Pulse {i} port single/double', ['Double', 'Single'])

            # Port Q
            self.create_quant(f'Pulse {i} port Q', 'Port Q', 'COMBO', group, section)
            self.combo_options([str(j) for j in range(1, 9)])
            self.visibility(f'Pulse {i} port single/double', ['Double'])

            # Type
            self.create_quant(f'Pulse {i} templates', 'Type', 'COMBO', group, section)
            self.combo_options(['Square', 'Long drive', 'Sample', 'Sin2', 'SinP', 'Triangle', 'Cool', 'Custom (coming soon™)'])
            self.visibility(f'Pulse {i} port single/double', ['Double', 'Single'])

            # SinP value
            self.create_quant(f'Pulse {i} sinP value', 'P', 'DOUBLE', group, section)
            self.visibility(f'Pulse {i} templates', ['SinP'])

    def section_templates(self):
        self.create_quant('Templates temp', 'D', 'STRING', '-_-_-_-_-', 'Custom templates')
        self.add_line('def_value: Not designed yet')
        self.add_line('permission: READ')

    def section_structure(self):
        section = 'Structure'
        group = 'General'
        self.small_comment('STRUCTURE GENERAL SETTINGS')

        # Averages
        self.create_quant('Average over', 'Average over', 'DOUBLE', group, section)
        self.add_line('low_lim: 1')

        # Iterations
        self.create_quant('Experiment iterations', 'Experiment iterations', 'DOUBLE', group, section)
        self.add_line('low_lim: 1')

        # Decay
        self.create_quant('Decay', 'Decay', 'DOUBLE', group, section)
        self.add_line('unit: s')
        self.add_line('tooltip: Time to wait between iterations')
        self.add_line('low_lim: 0')

        # Pulses per iter
        self.create_quant('Pulses per iter', 'Number of pulses per iteration', 'COMBO', group, section)
        self.combo_options([str(j) for j in range(1, 129)])

        group = 'Pulse order'
        self.small_comment('ORDER')
        for i in range(1, 129):
            self.create_quant(f'Output pulse {i}', f'{i} = pulse definition', 'DOUBLE', group, section)
            self.visibility('Pulses per iter', [str(j) for j in range(i, 129)])


Generator()
