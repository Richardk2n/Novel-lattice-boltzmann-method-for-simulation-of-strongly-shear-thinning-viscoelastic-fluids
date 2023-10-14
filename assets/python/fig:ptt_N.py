# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Sat Oct 14 19:57:01 2023

@author: Richard Kellnberger
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw
from style import *


def CY(eta0, etaInf, Wi, a1, a2):
    return etaInf + (eta0 - etaInf) / (1 + Wi**a1) ** (a2 / a1)


def ppt_eta(eta_p, Wi, epsilon):
    # tau_xxbygd = eta_p / (2 * epsilon * Wi) * lambertw(4 * epsilon * Wi**2)
    # return eta_p / np.exp(epsilon * Wi / eta_p * tau_xxbygd)
    return eta_p / np.exp(0.5 * lambertw(4 * epsilon * Wi**2))


def ppt_N(eta_p, gd, lam, epsilon):
    return eta_p / (2 * epsilon * lam) * lambertw(4 * epsilon * (lam * gd) ** 2)


def plot():
    wi = np.logspace(-4, 7, num=1000)
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(wi, ppt_N(100, wi, 1, 1), label=r"$\epsilon = 1$")
    plt.plot(wi, ppt_N(100, wi, 1, 0.1), label=r"$\epsilon = 0.1$")
    plt.plot(wi, ppt_N(100, wi, 1, 0.01), label=r"$\epsilon = 0.01$")
    plt.plot(wi, ppt_N(100, wi, 0.1, 1), label=r"$\epsilon = 1$")
    plt.plot(wi, ppt_N(100, wi, 0.1, 0.1), label=r"$\epsilon = 0.1$")
    plt.plot(wi, ppt_N(100, wi, 0.1, 0.01), label=r"$\epsilon = 0.01$")
    plt.plot(wi, ppt_N(1000, wi, 0.1, 1), label=r"$\epsilon = 1$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$Wi$")
    plt.ylabel(r"$\eta/\unit{\pascal\second}$")
    plt.yticks([1, 100], [r"$\eta_\infty$", r"$\eta_\text{p}$"])
    # plt.ylim(0.8, 125)
    plt.legend()
    # plt.savefig("../plots/ptt_N.eps")


if __name__ == "__main__":
    plot()
