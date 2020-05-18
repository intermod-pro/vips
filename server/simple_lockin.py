import math
import time

import numpy as np

from . import generate_lockin_stimuli as gls
from . import rflockin


class Lockin(object):
    _fs = 1e9
    _Nports = 8
    _Nfreqs = 32
    _Nchannels = 2
    _phasedepth = 2**40
    _max_DA_v = 1.
    _max_AD_v = 1.
    _max_DA_d = 2**15 - 1
    _max_AD_d = 2**15 - 1
    _max_amp_lo = 2**17 - 2  # -2 to avoid clipping when interpolating
    _max_amp_li = (2**17 - 1) * _max_AD_d

    def __init__(self):
        self.rflockin = rflockin.rflockin()

        self._ns = 2**20  # df approx 1 kHz
        self._freqs = np.zeros(self._Nfreqs, np.uint64)
        self._phases = np.zeros(
            (self._Nports + 1, self._Nfreqs, self._Nchannels), np.uint64)
        self._amps = np.zeros((self._Nports, self._Nfreqs), np.uint64)

    def __enter__(self):  # for use with 'with' statement
        return self

    def __exit__(self, exc_type, exc_value,
                 traceback):  # for use with 'with' statement
        self.close()

    def close(self):
        self.rflockin.close()

    def apply_settings(self):
        # compensate FPGA pipeline delays with phase offsets
        phases = np.copy(self._phases)
        for p in range(self._Nports+1):
            for f in range(self._Nfreqs):
                for c in range(self._Nchannels):
                    phases[p,f,c] += np.mod(-self._freqs[f]*f, self._phasedepth)

        # only enable channels with amplitude on at last one frequency
        enable = 1 # first bit is frequency generators for demodulators
        for p in range(self._Nports):
            if np.any(self._amps[p,:]):
                enable |= 2**(p+1)
        self.rflockin.enable_sources(enable)

        gls.setup(self._freqs, phases, self._amps)
        self.rflockin.output_control(2**32 - 1)
        self.rflockin.write_register(gls.list_of_commands)
        self.rflockin.write_register(gls.list_of_commands)  # TODO: remove and fix it!
        self.rflockin.output_control(0)

        self.rflockin.setup_lockin(self._ns // 2, self._ns // 2)

    def start_lockin(self):
        self.rflockin.start_lockin()

    def stop_lockin(self, empty_buffer=True):
        self.rflockin.stop_lockin()
        if empty_buffer:
            self.rflockin.empty_buffer(min_sleep=self.get_Tm())

    def tune_approx(self, f, df):
        ns = self._fs / df
        ns_tuned = int(round(ns / 8)) * 8
        df_tuned = self._fs / ns_tuned
        f_tuned = df_tuned * np.int64(np.round(f / df_tuned))
        return f_tuned, df_tuned

    def tune(self, f, df):
        ns = self._fs / df
        ns_tuned = 2**np.int64(np.round(np.log2(ns)))
        df_tuned = self._fs / ns_tuned
        f_tuned = df_tuned * np.int64(np.round(f / df_tuned))
        return f_tuned, df_tuned

    def get_fs(self):
        return self._fs

    def get_dt(self):
        fs = self.get_fs()
        return 1. / fs

    def get_df(self):
        return self._fs / self._ns

    def set_df(self, df):
        self._ns = int(round(self._fs / df))
        return self.get_df()

    def get_Tm(self):
        df = self.get_df()
        return 1. / df

    def set_Tm(self, Tm):
        df = 1. / Tm
        new_df = self.set_df(df)
        return 1. / new_df

    def get_ns(self):
        return self._ns

    def set_ns(self, ns):
        self._ns = int(round(ns))
        return self.get_ns()

    def get_frequencies(self):
        return self._freqs / 2 / self._phasedepth * self._fs

    def set_frequencies(self, freqs):
        freqs = np.atleast_1d(np.abs(freqs)).astype(np.float64)
        self._freqs.fill(0)
        self._freqs[:len(freqs)] = 2 * np.round(
            freqs / self._fs * self._phasedepth)

        self._resync_phases()
        return self.get_frequencies()

    def get_amplitudes(self, port=None):
        amps = self._amps / self._max_amp_lo * self._max_DA_v
        if port is None:
            return amps
        else:
            idx = port - 1
            return amps[idx, :]

    def set_amplitudes(self, amps, port=None):
        if port is None:
            amps = np.atleast_2d(np.abs(amps)).astype(np.float64)
        else:
            amps = np.atleast_1d(np.abs(amps)).astype(np.float64)
        if np.any(amps > self._max_DA_v):
            print("*** Warning: amplitude(s) clipped to maximum value!")
            amps[amps > self._max_DA_v] = self._max_DA_v
        _amps = np.int64(np.round(amps / self._max_DA_v * self._max_amp_lo))

        if port is None:
            self._amps.fill(0)
            self._amps[:, :_amps.shape[1]] = _amps
        else:
            idx = port - 1
            self._amps[idx, :] = 0
            self._amps[idx, :len(_amps)] = _amps

        return self.get_amplitudes(port=port)

    def _resync_phases(self):
        self._phases[:, :, 1] = np.mod(
            self._phases[:, :, 0] + (self._freqs // 2), self._phasedepth)

    def get_phases(self, port=None, deg=False):
        scale = 360. if deg else math.tau
        phases = self._phases / self._phasedepth * scale
        phases = np.mod(phases + scale / 2, scale) - scale / 2
        if port is None:
            return phases[1:, :, 0]  # skip demodulator and 2nd channel
        else:
            idx = port  # don't subtract 1: idx 0 is the demodulator, 1 is port 1, ...
            return phases[idx, :, 0]  # skip 2nd channel

    def set_phases(self, phases, port=None, deg=False):
        scale = 360. if deg else math.tau
        if port is None:
            phases = np.atleast_2d(phases).astype(np.float64)
        else:
            phases = np.atleast_1d(phases).astype(np.float64)
        phases = np.mod(phases, scale)
        _phases = np.int64(np.round(phases / scale * self._phasedepth))

        if port is None:
            self._phases.fill(0)
            self._phases[:, :_phases.shape[1], 0] = _phases
        else:
            idx = port  # don't subtract 1: idx 0 is the demodulator, 1 is port 1, ...
            self._phases[idx, :, 0] = 0
            self._phases[idx, :len(_phases), 0] = _phases

        self._resync_phases()
        return self.get_phases(port=port, deg=deg)

    def get_pixel(self, out=None):
        if out is None:
            out = np.empty((self._Nports, self._Nfreqs), np.complex128)
        _data = self.rflockin.get_pixel()
        data = _data / self.get_ns() / self._max_amp_li * self._max_AD_v
        out[:, :] = data[:, :, 0] + 1j * data[:, :, 1]
        return out

    def get_pixels(self, n):
        pixels = np.empty((n, self._Nports, self._Nfreqs), np.complex128)
        for ii in range(n):
            self.get_pixel(pixels[ii, :, :])
        return pixels

    def get_time_data(self, port, ns=None):
        if ns is None:
            ns = self.get_ns()
        idx = port - 1
        _data = self.rflockin.get_time_data(idx, ns)
        data = _data / self._max_AD_d * self._max_AD_v
        return data
