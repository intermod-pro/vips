import ctypes

import visa


class AgilentE8247C(object):
    def __init__(self):
        self.rm = visa.ResourceManager('@py')
        self.inst = self.rm.open_resource("TCPIP::192.168.18.104::INSTR")
        print(self.inst.query("*IDN?"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.inst.close()
        self.rm.close()

        self.inst = None
        self.rm = None

    def set_frequency(self, f):
        f = float(f)
        # if f < 250e3:
        #     raise ValueError("Minimum frequency is 9 kHz")
        # if f > 20e9:
        #     raise ValueError("Maximum frequency is 13 GHz")
        self.inst.write("FREQ {:.3f} HZ".format(f))

    def get_frequency(self):
        f_str = self.inst.query("FREQ?")
        return float(f_str.strip())

    def set_power(self, p):
        p = float(p)
        # if p < -20:
        #     raise ValueError("Minimum power is -20 dBm")
        # if p > 19:
        #     raise ValueError("Maximum power is +19 dBm")
        self.inst.write("POW {:.2f} dBm".format(p))

    def get_power(self):
        p_str = self.inst.query("POW?")
        return float(p_str.strip())

    def set_output(self, state):
        state = int(bool(state))
        self.inst.write("OUTP {:d}".format(state))

    def get_output(self):
        state_str = self.inst.query("OUTP?")
        return int(state_str.strip())

    def set_ext_ref(self, state):
        state = int(bool(state))
        self.inst.write("ROSC:SOUR:AUTO {:d}".format(state))

    def get_ext_ref(self):
        state_str = self.inst.query("ROSC:SOUR?")
        if state_str.strip().upper() == "EXT":
            return 1
        elif state_str.strip().upper() == "INT":
            return 0
        else:
            raise ValueError(state_str.strip())


class KeysightN5173B(object):
    def __init__(self):
        self.rm = visa.ResourceManager('@py')
        self.inst = self.rm.open_resource("TCPIP::192.168.18.106::INSTR")
        print(self.inst.query("*IDN?"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.inst.close()
        self.rm.close()

        self.inst = None
        self.rm = None

    def set_frequency(self, f):
        f = float(f)
        if f < 9e3:
            raise ValueError("Minimum frequency is 9 kHz")
        if f > 13e9:
            raise ValueError("Maximum frequency is 13 GHz")
        self.inst.write("FREQ {:.3f} HZ".format(f))

    def get_frequency(self):
        f_str = self.inst.query("FREQ?")
        return float(f_str.strip())

    def set_power(self, p):
        p = float(p)
        if p < -20:
            raise ValueError("Minimum power is -20 dBm")
        if p > 19:
            raise ValueError("Maximum power is +19 dBm")
        self.inst.write("POW {:.2f} dBm".format(p))

    def get_power(self):
        p_str = self.inst.query("POW?")
        return float(p_str.strip())

    def set_output(self, state):
        state = int(bool(state))
        self.inst.write("OUTP {:d}".format(state))

    def get_output(self):
        state_str = self.inst.query("OUTP?")
        return int(state_str.strip())

    def set_ext_ref(self, state):
        state = int(bool(state))
        self.inst.write("ROSC:SOUR:AUTO {:d}".format(state))

    def get_ext_ref(self):
        state_str = self.inst.query("ROSC:SOUR?")
        if state_str.strip().upper() == "EXT":
            return 1
        elif state_str.strip().upper() == "INT":
            return 0
        else:
            raise ValueError(state_str.strip())


class VaunixLMS(object):
    def __init__(self):
        self.vnx = ctypes.CDLL(
            r"F:\vnx_LMS_API\LMS64 SDK 6-16-18\vnx_fmsynth.dll")
        self.vnx.fnLMS_SetTestMode(0)
        nr_dev = self.vnx.fnLMS_GetNumDevices()
        if nr_dev < 1:
            raise RuntimeError("No device attached")
        elif nr_dev > 1:
            raise NotImplementedError(
                "More than one device attached, I don't know what to do")
        self._dev_arr = (ctypes.c_int * nr_dev)()
        self.vnx.fnLMS_GetDevInfo(self._dev_arr)
        self.dev = self._dev_arr[0]
        _name = (ctypes.c_char * 32)()
        _n = self.vnx.fnLMS_GetModelNameA(self.dev, _name)
        modelname = _name[:_n].decode()
        serial = self.vnx.fnLMS_GetSerialNumber(self.dev)
        print("Connected to Vaunix {:s} {:d}".format(modelname, serial))
        err = self.vnx.fnLMS_InitDevice(self.dev)
        if err:
            raise RuntimeError(
                "some error occurred when trying to initialize: {:d}".format(
                    err))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.vnx.fnLMS_CloseDevice(self.dev)

        self.dev = None
        self._dev_arr = None
        self.vnx = None

    def set_frequency(self, f):
        f = float(f)
        n = int(round(f / 10.))
        n_max = self.vnx.fnLMS_GetMaxFreq(self.dev)
        n_min = self.vnx.fnLMS_GetMinFreq(self.dev)
        if n < n_min:
            raise ValueError("Minimum frequency is 4 GHz")
        if n > n_max:
            raise ValueError("Maximum frequency is 8 GHz")
        self.vnx.fnLMS_SetFrequency(self.dev, n)

    def get_frequency(self):
        n = self.vnx.fnLMS_GetFrequency(self.dev)
        f = 10 * n
        return float(f)

    def set_power(self, p):
        p = float(p)
        n = int(round(p * 4))
        n_max = self.vnx.fnLMS_GetMaxPwr(self.dev)
        n_min = self.vnx.fnLMS_GetMinPwr(self.dev)
        if n < n_min:
            raise ValueError("Minimum power is -40 dBm")
        if n > n_max:
            raise ValueError("Maximum power is +10 dBm")
        self.vnx.fnLMS_SetPowerLevel(self.dev, n)

    def get_power(self):
        n = self.vnx.fnLMS_GetAbsPowerLevel(self.dev)
        p = 0.25 * n
        return p

    def set_output(self, state):
        state = int(bool(state))
        self.vnx.fnLMS_SetRFOn(self.dev, state)

    def get_output(self):
        state = self.vnx.fnLMS_GetRF_On(self.dev)
        return state

    def set_ext_ref(self, state):
        int_state = int(not state)
        self.vnx.fnLMS_SetUseInternalRef(self.dev, int_state)

    def get_ext_ref(self):
        int_state = self.vnx.fnLMS_GetUseInternalRef(self.dev)
        ext_state = not int_state
        return ext_state
