# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Mon Jan 29 15:21:55 2024

@author: Richard Kellnberger
"""

import numpy as np
from numpy.fft import fft, fftfreq
from scipy.optimize import curve_fit

from fluidx3d.eval.models.Roscoe import modelRadiusSquared, modelX1, modelY1


def wrapModelX(alpha1, alpha2, nu, phase):
    def modelX(t, theta):
        return modelX1(t, alpha1, alpha2, theta, nu, phase)

    return modelX


def wrapModelY(alpha1, alpha2, nu, phase):
    def modelY(t, theta):
        return modelY1(t, alpha1, alpha2, theta, nu, phase)

    return modelY


def getModelTheta(alpha1s, alpha2s, nu, phase):
    modelX = wrapModelX(np.sqrt(alpha1s), np.sqrt(alpha2s), nu, phase)
    modelY = wrapModelY(np.sqrt(alpha1s), np.sqrt(alpha2s), nu, phase)

    def modelTheta(t, theta):
        return np.concatenate((modelX(t, theta), modelY(t, theta)))

    return modelTheta


def getInitialAlphas(point1):
    a1s = np.max(point1[..., 0] ** 2 + point1[..., 1] ** 2)
    a2s = np.min(point1[..., 0] ** 2 + point1[..., 1] ** 2)
    return a1s, a2s


def getInital(point1, timesteps):
    a1sGuess, a2sGuess = getInitialAlphas(point1)
    cosNuTs = (point1[..., 0] ** 2 + point1[..., 1] ** 2 - a2sGuess) / (a1sGuess - a2sGuess)

    transformed = fft(cosNuTs)
    freq = fftfreq(timesteps.shape[-1], d=timesteps[1] - timesteps[0])

    # The square gives double frequency -> only multiply with pi
    return a1sGuess, a2sGuess, -np.abs(freq[np.argmin(transformed)]) * np.pi


# Expects points to be centered. Expect normalized.
def extractRoscoe(timesteps, point1, point5, cutoff):
    t = timesteps[cutoff:]
    p1 = point1[cutoff:]
    p5 = point5[cutoff:]
    a1sGuess, a2sGuess, ttfGuess = getInital(p1, t)

    param1, cov1 = curve_fit(
        modelRadiusSquared,
        t,
        p1[..., 0] ** 2 + p1[..., 1] ** 2,
        [a1sGuess, a2sGuess, ttfGuess, np.pi],
        bounds=([1, 0, -np.inf, 0], [np.inf, np.inf, 0, 2 * np.pi]),
    )
    e1 = np.sqrt(np.diag(cov1))

    alpha3 = np.average(p5[..., 2])
    err3 = np.std(p5[..., 2], ddof=1)

    param2, cov2 = curve_fit(
        getModelTheta(*param1),
        t,
        np.concatenate((p1[..., 0], p1[..., 1])),
        [np.pi / 8],
        bounds=([0], [np.pi / 4]),
    )
    e2 = np.sqrt(np.diag(cov2))

    a1s, a2s, ttf, phase = param1
    a1 = np.sqrt(a1s)
    a2 = np.sqrt(a2s)
    err1s, err2s, errttf, errPhase = e1
    err1 = err1s / (2 * a1)
    err2 = err2s / (2 * a2)
    a3 = alpha3
    th = param2[0]
    errth = e2[0]

    return (a1, a2, a3, th, ttf, phase), (err1, err2, err3, errth, errttf, errPhase)


__all__ = ["extractRoscoe"]
