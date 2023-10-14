# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Sat Oct 14 19:57:01 2023

@author: Richard Kellnberger
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from style import *


def CY(eta0, etaInf, Wi, a1, a2):
    return etaInf + (eta0 - etaInf) / (1 + Wi**a1) ** (a2 / a1)


def plot():
    wi = np.logspace(-4, 7, num=1000)
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(wi, CY(100, 1, wi, 1, 1), label=r"$a_1 = 1$")
    plt.plot(wi, CY(100, 1, wi, 2, 1), label=r"$a_1 = 2$")
    plt.plot(wi, CY(100, 1, wi, 3, 1), label=r"$a_1 = 3$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$Wi$")
    plt.ylabel(r"$\eta/\unit{\pascal\second}$")
    plt.yticks([1, 100], [r"$\eta_\infty$", r"$\eta_0$"])
    plt.legend()
    plt.savefig("../plots/CY.eps")


if __name__ == "__main__":
    plot()
