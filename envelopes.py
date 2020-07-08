# Authored by Johan Blomberg and Gustav Grännsjö, 2020

"""
A collection of functions for generating specific envelope shapes.
"""

import numpy as np
from scipy.stats import norm


def sin2(nr_samples):
    x = np.linspace(0.0, 1.0, nr_samples)
    return np.sin(np.pi * x)**2


def sin_p(p, nr_samples):
    x = np.linspace(0.0, 1.0, nr_samples)
    return np.sin(np.pi * x)**p


def sinc(lim, nr_samples):
    x = np.linspace(-lim, lim, nr_samples)
    return np.sinc(x)


def triangle(nr_samples):
    if nr_samples % 2 == 0:
        t1 = np.linspace(0.0, 1.0, nr_samples // 2)
        t2 = np.linspace(1.0, 0.0, nr_samples // 2)
        return np.concatenate((t1, t2))
    else:
        t1 = np.linspace(0.0, 1.0, nr_samples // 2, endpoint=False)
        t2 = np.flip(t1)
        return np.concatenate((t1, [1], t2))


def cool(nr_samples):
    x = np.linspace(0.0, 1.0, nr_samples)
    s = np.sin(4 * np.pi * x)
    t = triangle(nr_samples)
    return t * s


def gaussian(nr_samples, trunc):
    x = np.linspace(-trunc, trunc, nr_samples)
    y = norm.pdf(x, 0, 1)
    # Normalise
    return y / y.max()
