import json
import os
import sys
import time

import numpy as np

# *** See below for code that sets the save_folder ***
my_path = os.path.realpath(__file__)
my_folder = os.path.dirname(my_path)
paths_path = os.path.join(my_folder, "paths.json")
try:
    with open(paths_path, mode="rt", encoding="utf-8") as f:
        _save_folder = json.load(f)
except FileNotFoundError:
    _save_folder = my_folder
    with open(paths_path, mode="wt", encoding="utf-8") as f:
        json.dump(_save_folder, f)


def write_save_folder(save_folder):
    global _save_folder
    _save_folder = save_folder
    with open(paths_path, mode="wt", encoding="utf-8") as f:
        json.dump(save_folder, f)


def get_load_path(script_filename, save_folder=None):
    if save_folder is None:
        save_folder = _save_folder

    scriptname = os.path.splitext(os.path.basename(script_filename))[0]
    if scriptname.startswith("load_"):
        scriptname = scriptname[5:]
    prefix = "".join([scriptname, "_20"])
    suffix = ".npz"

    try:
        load_filename = sys.argv[1]
    except Exception:
        print("No or invalid filename specified...")

        all_files = sorted(os.listdir(save_folder))
        my_files = [
            x for x in all_files if x.startswith(prefix) and x.endswith(suffix)
        ]
        if my_files:
            load_filename = my_files[-1]
            print("Falling back to {:s}".format(load_filename))
        else:
            print("No valid file found")
            sys.exit()

    load_path = os.path.join(save_folder, load_filename)
    return load_path


def format_sec(s):
    """ Utility function to format a time interval in seconds
    into a more human-readable string.

    Args:
        s (float): time interval in seconds

    Returns:
        (str): time interval in the form "Xh Ym Z.zs"

    Examples:
        >>> format_sec(12345.6)
        '3h 25m 45.6s'
    """
    if s < 1.:
        return "{:.1f}ms".format(s * 1e3)

    h = int(s // 3600)
    s -= h * 3600.

    m = int(s // 60)
    s -= m * 60

    if h:
        res = "{:d}h {:d}m {:.1f}s".format(h, m, s)
    elif m:
        res = "{:d}m {:.1f}s".format(m, s)
    else:
        res = "{:.1f}s".format(s)

    return res


def get_savepath(script_filename, save_folder=None, struct_time=None):
    if save_folder is None:
        save_folder = _save_folder
    if struct_time is None:
        struct_time = time.localtime()
    scriptname = os.path.splitext(os.path.basename(script_filename))[0]
    save_filename = "{:s}_{:s}.npz".format(
        scriptname,
        time.strftime("%Y%m%d_%H%M%S", struct_time),
    )
    save_path = os.path.join(save_folder, save_filename)
    return save_path


def get_sourcecode(script_filename):
    with open(script_filename, mode='rt', encoding='utf-8') as f:
        sourcecode = f.readlines()
    return sourcecode


def untwist_downconversion(I_port, Q_port):
    L_sideband = np.zeros_like(I_port)
    H_sideband = np.zeros_like(Q_port)

    L_sideband.real += I_port.real - Q_port.imag
    L_sideband.imag += Q_port.real + I_port.imag
    H_sideband.real += I_port.real + Q_port.imag
    H_sideband.imag += Q_port.real - I_port.imag

    L_sideband.imag *= -1
    H_sideband.imag *= -1

    return L_sideband, H_sideband


def sin2(nr_samples):
    x = np.linspace(0.0, 1.0, nr_samples, endpoint=False)
    return np.sin(np.pi * x)**2


def sinP(P, nr_samples):
    x = np.linspace(0.0, 1.0, nr_samples, endpoint=False)
    return np.sin(np.pi * x)**P


def triangle(nr_samples):
    t1 = np.linspace(0.0, 1.0, nr_samples // 2, endpoint=False)
    t2 = np.linspace(1.0, 0.0, nr_samples // 2, endpoint=False)
    return np.concatenate((t1, t2))


def cool(nr_samples):
    x = np.linspace(0.0, 1.0, nr_samples, endpoint=False)
    s = np.sin(4 * np.pi * x)
    t = triangle(nr_samples)
    return t * s


def demodulate(signal, nc, bw=None):
    if bw is None:
        bw = nc
    ns = len(signal)
    s_fft = np.fft.rfft(signal) / ns
    e_fft = np.zeros(bw, dtype=np.complex128)
    karray = np.arange(nc - bw // 2, nc + bw // 2 + 1)
    e_fft[karray - nc] = s_fft[karray]
    envelope = np.fft.ifft(e_fft) * bw
    return envelope


def demodulate_time(t, bw):
    t0 = t[0]
    dt = t[1] - t[0]
    t1 = t[-1] + dt
    t_e = np.linspace(t0, t1, bw, endpoint=False)
    return t_e
