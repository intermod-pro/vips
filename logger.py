# Authored by Johan Blomberg and Gustav Grännsjö, 2020

import os
from pathlib import Path
import time


class Logger:
    """
    ▗▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▖ DEBUG ▗▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▞▚▖
    """

    """
    Debug mode is enabled with debug_enable = True. In Debug mode, every call to a Vivace method
    will be written to a log file. This lets the developer see if the given set of instrument
    settings correctly translate to the desired functionality.
    The file is written to C:/Users/[username]/DEBUG_PATH/debug_file_name. If no
    file name is given, it will default to 'log.txt'.
    """
    enable = False
    overwrite = True
    file_name = ''
    working_file_name = ''
    new_log = True
    USER_DIR = os.path.expanduser('~')
    DEBUG_PATH = 'Vivace_Sequencer_Debug'
    INITIAL_TIME = None

    def add_line(self, string):
        """
        DEBUG:
        Add the given string as a line to the string that will be written to the log file.
        Also writes the current version of that string to the log file.
        """
        if self.enable:
            if self.new_log:
                self.initialise_log_file()
                self.new_log = False

            if self.INITIAL_TIME is None:
                self.INITIAL_TIME = time.time()

            directory = os.path.join(self.USER_DIR, self.DEBUG_PATH)
            with open(os.path.join(directory, f'{self.working_file_name}.txt'), 'a') as f:
                f.write(str(time.time() - self.INITIAL_TIME) + ": " + string + '\n')

    def initialise_log_file(self):
        """
        DEBUG:
        Empty the log file.
        """
        if self.enable:
            filename = self.file_name
            directory = os.path.join(self.USER_DIR, self.DEBUG_PATH)
            Path(directory).mkdir(parents=True, exist_ok=True)
            full_path = os.path.join(directory, f'{filename}.txt')

            # If not in overwrite mode, add a unique suffix to log name
            if not self.overwrite:
                suffix = 1
                while Path(full_path).is_file():
                    filename = self.file_name + f'_{suffix}'
                    full_path = os.path.join(directory, f'{filename}.txt')
                    suffix += 1
                self.working_file_name = filename

            with open(full_path, 'w') as f:
                f.write('')

    def print_trigger_sequence(self, q):
        """
        DEBUG:
        Print out the low-level instructions that are set up in Vivace.
        """
        if self.enable:
            for trigger in q.seq:
                parts = str(trigger).split(',', 1)
                self.add_line(parts[1])
