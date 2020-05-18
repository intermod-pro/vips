import select
import socket
import struct
import time
import numpy as np

COMMAND_START_LOCKIN = 1
COMMAND_STOP_LOCKIN = 2
COMMAND_START_TIME = 3
COMMAND_OUTPUT_CONTROL = 4
COMMAND_CONTROL_LOCKIN2 = 6
COMMAND_WRITE_REGISTER = 7
COMMAND_START_QUANTUM = 8
COMMAND_STOP_QUANTUM = 9
COMMAND_START_Q_AVERAGE = 10
COMMAND_STOP_Q_AVERAGE = 11
COMMAND_GET_BUFFER = 12
COMMAND_GET_REGISTER = 13
COMMAND_SETUP_LOCKIN = 14
COMMAND_SET_DAC_AVTT = 15


class rflockin:
    def __init__(self):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

        # Connect the socket to the port where the server is listening
        server_address = ('192.168.42.50', 3490)
        print('connecting to %s port %s' % server_address)
        self.sock.connect(server_address)

        self.set_dac_avtt(3.0)  # TODO: do it better!

    def __enter__(self):  # for use with 'with' statement
        return self

    def __exit__(self, exc_type, exc_value,
                 traceback):  # for use with 'with' statement
        self.close()

    def __del__(self):  # get rid of this?
        self.close()

    def close(self):
        if self.sock is None:
            print('no open socket')
        else:
            # self.stop_lockin()
            time.sleep(.1)  # get rid of this
            print('closing socket')
            self.sock.close()
            self.sock = None

    def setup_lockin(self, nsamples_adc, nsamples_dac):
        message = struct.pack('<IIII', 16, COMMAND_SETUP_LOCKIN,
                              nsamples_adc, nsamples_dac)
        self.sock.sendall(message)

    def start_lockin(self):
        message = struct.pack('<II', 8, COMMAND_START_LOCKIN)
        self.sock.sendall(message)

    def stop_lockin(self):
        message = struct.pack('<II', 8, COMMAND_STOP_LOCKIN)
        self.sock.sendall(message)

    def empty_buffer(self, min_sleep=None, verbose=False):
        if min_sleep is None:
            sleep_time = 0.001
        else:
            sleep_time = max(0.001, min_sleep)
        if verbose:
            print("Emptying buffer...", end="")
        input_list = [self.sock]
        first_empty = True
        count = 0
        while True:
            inputready, o, e = select.select(input_list, [], [], 0.0)
            if len(inputready) == 0:
                if first_empty:
                    # try again to be sure
                    if verbose:
                        print(" -", end="")
                    first_empty = False
                    time.sleep(sleep_time)
                else:
                    break
            for s in inputready:
                if not first_empty:
                    if verbose:
                        print(" +", end="")
                    first_empty = True
                s.recv(8 * 32 * 16)
                count += 1
        if verbose:
            print(" Done. Removed {:d} packets.".format(count))

    def output_control(self, mask):
        message = struct.pack('<III', 12, COMMAND_OUTPUT_CONTROL, mask)
        self.sock.sendall(message)

    def get_pixel(self):
        data = self.sock.recv(4)
        amount_received = len(data)
        packet = data
        amount_expected = 8 * 32 * 16
        while amount_received < amount_expected:
            data = self.sock.recv(amount_expected - amount_received)
            packet += data
            amount_received += len(data)
        return np.frombuffer(packet,
                             dtype=np.int64).reshape(8, 32,
                                                     2).astype(np.float64)

    def get_time_data(self, channel, number_of_samples):
        message = struct.pack('<IIII', 16, COMMAND_START_TIME, channel,
                              number_of_samples)
        self.sock.sendall(message)
        b = bytearray()
        while len(b) < number_of_samples * 2:
            b += self.sock.recv(number_of_samples * 2 - len(b))
        return np.frombuffer(b, np.int16)

    def control_lockin2(self, P, I, S):
        L = list(('<II' + 'Q' * (1 + 32 * 3), ))
        L += list((4 + 4 + 8 * (32 * 3 + 1), ))
        L += list((COMMAND_CONTROL_LOCKIN2, ))
        L += list((0, ))
        L += P + I + S
        message = struct.pack(*L)
        self.sock.sendall(message)

    def increment_from_hz(self, x):
        i = 2**40 * x / 1e9  # 1 GHz domain
        i *= 2  # 500 MHz domain
        if int(i) != i:
            print("Round!!!")
        return int(i)

    def AD_to_V(self, x):
        return x / 2**15 * 0.5914010499526942

    def V_to_DA(self, x):
        return round(x / (0.4523 / 2) * (2**17 - 2))

    def start_quantum(self):
        message = struct.pack('<II', 8, COMMAND_START_QUANTUM)
        self.sock.sendall(message)

    def stop_quantum(self):
        message = struct.pack('<II', 8, COMMAND_STOP_QUANTUM)
        self.sock.sendall(message)

    def write_register(self, l):
        flat = sum(l, [])
        n = len(flat)
        s = '<II' + str(n) + 'Q'
        message = struct.pack(s, 8 + 8 * n, COMMAND_WRITE_REGISTER, *flat)
        self.sock.sendall(message)

    def _receive(self, N):
        data = self.sock.recv(4)
        amount_received = len(data)
        packet = data
        amount_expected = N
        while amount_received < amount_expected:
            data = self.sock.recv(amount_expected - amount_received)
            packet += data
            amount_received += len(data)
        return packet

    def get_chunk(self, N):
        packet = self._receive(N)
        return np.frombuffer(packet, dtype=np.int16)

    def start_average_quantum(self, N):
        message = struct.pack("<III", 8 + 4, COMMAND_START_Q_AVERAGE, N)
        self.sock.sendall(message)

    def stop_average_quantum(self):
        message = struct.pack('<II', 8, COMMAND_STOP_Q_AVERAGE)
        self.sock.sendall(message)

    def read_buffer(self, N):
        message = struct.pack("<III", 8 + 4, COMMAND_GET_BUFFER, N)
        self.sock.sendall(message)
        packet = self._receive(N)
        return np.frombuffer(packet, dtype=np.int32)

    def read_register(self, N):
        message = struct.pack("<III", 8 + 4, COMMAND_GET_REGISTER, N)
        self.sock.sendall(message)
        packet = self._receive(8)
        return np.frombuffer(packet, dtype=np.uint64)[0]

    def enable_sources(self, mask):
        self.write_register([[20*2, mask]])

    def set_dac_avtt(self, voltage):
        if voltage == 2.5:
            volt_int = 2
        elif voltage == 3.0:
            volt_int = 3
        else:
            raise ValueError("Unsupported voltage {} -- choose 2.5 or 3.0".format(voltage))
        message = struct.pack('<III', 12, COMMAND_SET_DAC_AVTT, volt_int)
        self.sock.sendall(message)
