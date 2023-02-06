# Authored by Johan Blomberg and Gustav Grännsjö, 2020

class Generator:
    """
    Class for building a Labber Driver INI file.

    Version 1.1

    The methods in this class add lines to a string variable,
    which can then be written to a file with the write() method once
    it contains everything one wants the INI file to contain.

    The lines are added in the order you call methods. The general workflow is to call create_quant()
    to set up a quant, then call some additional helper methods to add
    specific settings to that quant. Once you are done with the quant,
    simply call create_quant() again to move on to a new one.
    """
    contents = ''

    def big_comment(self, string):
        """
        Output a three line block comment of octothorpes with the given string in the middle.
        """
        self.contents += """
###############################
### {}
###############################
""".format(string + ' ' + '#'*(26-len(string)))

    def small_comment(self, string):
        """
        Output a one line comment with octothorpes to the left and right of it.
        """
        self.contents += '### {}\n'.format(string + ' ' + '#'*(26-len(string)))

    def general_settings(self, name, version, driver_path, author='', signal_generator=False, signal_analyzer=False, interface='None'):
        """
        Outputs the header for the [General settings] along with a few
        important settings.
            name: the name of your instrument
            version: version number of your instrument driver
            driver_path: the name of the folder in your driver directory that your INI is in
            author: author information
            signal_generator: whether this instrument is a signal generator. Defaults to False
            signal_analyzer: whether this instrument is a signal analyzer. Defaults to False
            interface: default communication interface used by this instrument
        """
        self.contents += f"""# Authored by {author}

[General settings]

# The name is shown in all the configuration windows
name: {name}

# The version string
version: {version}

# Name of folder containing the code defining a custom driver
driver_path: {driver_path}

signal_generator: {signal_generator}

signal_analyzer: {signal_analyzer}

interface: {interface}
"""


    def create_quant(self, name, label, datatype, group, section):
        """
        Output the basic settings for a quant.
        """
        self.contents += f"""
[{name}]
label: {label}
datatype: {datatype}
group: {group}
section: {section}
"""

    def combo_options(self, *options):
        """
        Output any number of options for a COMBO box quant.
        Options uses the vararg format; enter values comma-separated or as a list preceded by *
        """
        string = ''
        for i, o in enumerate(options):
            string += f'combo_def_{i+1}: {o}\n'

        self.contents += string

    def visibility(self, quant, *values):
        """
        Used to make the current quant only visible when the given quant has any of the given values.
        Values uses the vararg format; enter values comma-separated or as a list preceded by *
        """
        string = f'state_quant: {quant}\n'

        for i, v in enumerate(values):
            string += f'state_value_{i+1}: {v}\n'

        self.contents += string

    def limits(self, low=None, high=None):
        """
        Set upper and lower limits for the current quant. Leave blank or enter None to not set
        a limit.
        """
        string = ''
        if low is not None:
            string += f'low_lim: {low}\n'
        if high is not None:
            string += f'high_lim: {high}\n'
        self.contents += string

    def default(self, value):
        """
        Set a default value for the current quant.
        """
        self.contents += f'def_value: {value}\n'

    def unit(self, value):
        """
        Set a unit for the current quant.
        """
        self.contents += f'unit: {value}\n'

    def tooltip(self, tip):
        """
        Set a tooltip for the current quant.
        """
        self.contents += f'tooltip: {tip}\n'

    def permission(self, perm):
        """
        Set a permission for the current quant. Options are READ, WRITE, BOTH, NONE.
        """
        self.contents += f"permission: {perm}\n"

    def set_cmd(self, *commands):
        """
        Set set_cmd values for the current quant.
        If you enter multiple values into *commands, they will be written comma-separated
        in the INI.
        """
        string = f'set_cmd: {commands[0]}'
        for cmd in commands[1:]:
            string += ', ' + cmd
        self.contents += string + '\n'

    def get_cmd(self, *commands):
        """
        Set get_cmd values for the current quant.
        If you enter multiple values into *commands, they will be written comma-separated
        in the INI.
        """
        string = f'get_cmd: {commands[0]}'
        for cmd in commands[1:]:
            string += ', ' + cmd
        self.contents += string + '\n'

    def show_in_measurement(self, value):
        """
        Set the show_in_measurement_dlg value for the current quant.
        When True, the quant in question will show up by default in the measurement editor.
        Note that as long as this value is True for any quant, ONLY quants with
        this value set to True will show up in the measurement editor by default.
        """
        self.contents += f'show_in_measurement_dlg: {value} \n'

    def add_line(self, string):
        """
        Add the given string as a new line into the INI.
        Use this for all situations not covered by any of the other methods.
        """
        self.contents += string + '\n'

    def write(self, filename):
        """
        Write the accumulated lines into a file with the given file name.
        Call this at the end of your script.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.contents)
