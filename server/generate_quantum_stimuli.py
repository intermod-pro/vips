import numpy as np

class generate_quantum_stimuli(object):
    _loreg = 0
    _hireg = 1
    _control = 0
    _inc = 2**63
    _pha = 2**62
    _sca = 2**61

    def __init__(self):
        self._quantum_list_of_commands = []

    def __enter__(self):
        return self

    def print_commands(self):
        for a in self._quantum_list_of_commands:
            r = a[0] * 8
            d = a[1]
            print("{:010x} {:016x}".format(r, d))

    def printreg(self, r, hireg, d):
        assert int(r) == r, "printreg, register not int"
        assert int(hireg) == hireg, "printreg, hireg not int"
        assert int(d) == d, "printreg, data not int"
        self._quantum_list_of_commands.append([r * 2 + hireg, d])


    def setup_template(self, template_index, data):
        # print("setup_template {} {}".format(template_index,data))
        data = np.array(data, dtype=np.uint16)
        #wid
        self.printreg(128, self._loreg, (2**template_index) & (2**64 - 1))
        self.printreg(128, self._hireg, (2**template_index) >> 64)

        dr = data.reshape(-1, 8)
        #data
        for c, d in enumerate(dr):
            # address and write enable to control
            self.printreg(130, self._loreg, 2**16 + c)
            # data
            self.printreg(
                129, self._loreg,
                int(d[0]) + (int(d[1]) << 16) + (int(d[2]) << 32) +
                (int(d[3]) << 48))
            self.printreg(
                129, self._hireg,
                int(d[4]) + (int(d[5]) << 16) + (int(d[6]) << 32) +
                (int(d[7]) << 48))

        #remove write enable
        self.printreg(130, self._loreg, 0)

        #set data
        self.printreg(129, self._loreg, ((dr.shape[0] - 2) << 16) + 1)

        #strobe control enable
        self.printreg(130, self._loreg, 2**17)
        self.printreg(130, self._loreg, 0)


    def setup_freq_lut(self, template_index, increment, phase, addr_inc):
        #wid
        self.printreg(131, self._loreg, (2**template_index) & (2**64 - 1))
        self.printreg(131, self._hireg, (2**template_index) >> 64)

        #data
        for i in range(len(increment)):
            # address and write enable to control
            self.printreg(133, self._loreg, 2**16 + i)
            # data
            self.printreg(132, self._loreg, increment[i])
            self.printreg(132, self._hireg, phase[i])

        #remove write enable
        self.printreg(133, self._loreg, 0)

        #set data
        self.printreg(132, self._loreg, ((len(increment)*(2**10)-addr_inc) << 32) + addr_inc)

        #strobe control enable
        self.printreg(133, self._loreg, 2**17)
        self.printreg(133, self._loreg, 0)


    def setup_scale_lut(self, template_index, data, addr_inc):
        #wid
        self.printreg(131, self._loreg, 2**(template_index+8) & (2**64 - 1))
        self.printreg(131, self._hireg, 2**(template_index+8) >> 64)

        #data
        for i in range(len(data)):
            # address and write enable to control
            self.printreg(133, self._loreg, 2**16 + i)
            # data
            self.printreg(132, self._loreg, data[i])

        #remove write enable
        self.printreg(133, self._loreg, 0)

        #set data
        self.printreg(132, self._loreg, ((len(data)*(2**10) - addr_inc) << 32) + addr_inc)

        #strobe control enable
        self.printreg(133, self._loreg, 2**17)
        self.printreg(133, self._loreg, 0)


    def write_trigger_word(self, address, data):
        ce = 1
        d = data
        for i in range(18):
            self.printreg(144, self._loreg, (ce << 46) | (address << 32) | ((d&0xffffffff) << 0))
            d >>= 32
            ce <<= 1


    def write_trigger(self, address, mask, t, stop, restart):
        #print(address, bin(mask), t, stop, restart)
        data = restart
        data <<= 1
        data |= stop
        data <<= 32
        data |= t
        data <<= 512
        data |= mask
        self.write_trigger_word(address, data)


    def generate_mask(self, l):
        mask = 0
        for i in l:
            if isinstance(i, tuple):
                mask |= 1 << (i[0] * 16 + i[1])
            else:
                mask |= 1 << i
        return mask

    def output_index(self, i, j):
        return int(i * 16 + j)


    def store_index(self, i):
        return int(128 + i)


    def readout_index(self):
        return 136

    def wavegen_index(self, i):
        return int(137+i)

    def freq_index(self, i):
        return int(145+i)

    def scale_index(self, i):
        return int(153+i)

    def use_scale_index(self, i):
        return int(161+i)




    def setup_measurement(self, period, repeat_count, triggers):
        """ Create register writes to set up a measurement sequence.

        Parameters:
        period : int
            The total period of the measurement sequence.
        repeat_count : int
            Number of times to repeat the measurement sequence.
        triggers : list (start:int, len:int, trigger : int)

        """
        tnone = (2**32) - 1  # dummy time that will never be reached
        count = 0
        state = 0

        # split pulses into on and off events
        command_list = []
        for i in triggers:
            command_list.append([i[0], 1 << i[2]])
            command_list.append([i[0] + i[1], 1 << i[2]])
        command_list.sort()

        # print("After split:")
        # for e in command_list:
        #     print(e[0], hex(e[1]))

        # join
        joined_command_list = [command_list.pop(0)]
        for e in command_list:
            if e[0] == joined_command_list[-1][0]:
                joined_command_list[-1][1] ^= e[1]
            else:
                joined_command_list.append(e)

        # print("After join:")
        # for e in joined_command_list:
        #     print(e[0], hex(e[1]))

        # generate control words
        for e in joined_command_list:
            self.write_trigger(count, state, e[0], 0, 0)
            state ^= e[1]
            count += 1

        # restart trig
        self.write_trigger(count, 0, period - 3, 0, 0)
        count += 1
        self.write_trigger(count, 0, period - 2, 0, 1)
        count += 1

        # do nothing waiting for restart to apply
        self.write_trigger(count, 0, tnone, 0, 0)
        count += 1

        # remove write enable
        self.printreg(144, self._loreg, 0)

        # write repeat count register
        self.printreg(144, self._hireg, repeat_count)

        return count


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if 0:
        # setup some templates
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)

        d = np.sin(theta) * 32767 / 2
        d = np.concatenate((np.zeros(8), d))
        setup_template(0, d.astype(int))

        d = np.sin(2 * theta) * 32767 / 2
        d = np.concatenate((np.zeros(8), d))
        setup_template(16, d.astype(int))

        trig = [
            [500, 32, output_index(0, 0)],
            [600, 32, output_index(1, 0)],
            [600, 32, store_index(1)],
            [700, 1, readout_index()],
            [1500, 32, output_index(0, 0)],
            [1800, 32, output_index(1, 0)],
            [1800, 32, store_index(1)],
            [2000, 1, readout_index()],
        ]

        setup_measurement(3000, 5, trig)

        #withdraw fifo reset
        self.printreg(1, loreg, 8)

        # start dma
        # self.printreg(2, loreg, 0x0000000010000000);
        # self.printreg(3, loreg, 0xC000000010002000);
        # two readouts, each 32(cycles)*8(samples/cycle)*16(bytes) = 4096
        self.printreg(8, loreg, 0x0000000000000000)
        self.printreg(9, loreg, 0xC000000000004000)
        self.printreg(16, loreg, 0x0000000000000000)
        self.printreg(17, loreg, 0xC000000000004000)

        #trigger enable
        self.printreg(1, loreg, 1 + 4 + 8)

        print_commands()

    # setup parameter luts and iterate through them
    if 0:
        setup_freq_lut(0,[2**35,2**36,2**37,2**38], [2**38+2**39]*4, 2**10)
        setup_scale_lut(0,[2**16,2**15,2**14,2**13], 2**8)
        setup_template(output_index(0,0), np.concatenate((np.zeros(8), np.linspace(0, 32767, 8*32))))
        setup_template(output_index(0,1), np.array([0]*8+[32767]*16, dtype=int))
        setup_template(output_index(0,2), np.concatenate((np.zeros(8), np.linspace(32767, 0, 8*32))))
        trig = [
            [20, 32*3, wavegen_index(0)],
            [20+32*0, 32, output_index(0,0)],
            [20+32*1, 32, output_index(0,1)],
            [20+32*2, 32, output_index(0,2)],
            [400, 1, freq_index(0)],
            [400, 1, scale_index(0)],
        ]
        setup_measurement(500, 16, trig)

        # modulate templates
        self.printreg(19, loreg, 2**64-1)
        #trigger enable
        self.printreg(1, loreg, 1 + 2 + 4 + 8 + 16 + 32)
        print_commands()

    if 1:
        N = 32
        z = np.zeros(8)
        pos = 32767 * np.blackman(N*8)
        neg = -pos * 0.5
        g = generate_quantum_stimuli()
        g.setup_freq_lut(0,[2**42,], [0,]*4, 0)
        g.setup_scale_lut(0,[2**16], 0)
        g.setup_template(g.output_index(0,0), np.concatenate((z, pos)))
        g.setup_template(g.output_index(0,1), np.concatenate((z, neg)))
        trig = [
            [20, N, g.wavegen_index(0)],
            [20, N, g.output_index(0,0)],
            # [20, N, g.output_index(0,1)],
        ]
        g.setup_measurement(500, 1, trig)

        # modulate templates
        g.printreg(19, g._loreg, 2**64-1)
        #trigger enable
        g.printreg(1, g._loreg, 1 + 2 + 4 + 8 + 16 + 32)
        g.print_commands()

    # trigger_enable   <= reset_reg_sample_clk(0);
    # samplefifo_rst   <= not reset_reg_sample_clk(1);
    # use_averaging    <= reset_reg_sample_clk(2);
    # fifo_ri(7).reset <= not reset_reg_sample_clk(3);
    # fifo_ri(8).reset <= not reset_reg_sample_clk(3);
    # reset_plut       <= not reset_reg_sample_clk(4);
