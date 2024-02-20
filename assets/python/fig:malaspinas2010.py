# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from fluidx3d.eval.style import cm, color

R = 10e-6


# https://doi.org/10.1016/j.jnnfm.2010.09.001 eq. 57 and 61 corrected
def u(r, vmax):
    return (1 - r**2 / R**2) * vmax


def process(index):
    VmaxSI = 1
    if index >= 8:
        VmaxSI = 0.14285714285714285
    baseDir = Path(f"../data/malaspinas2010/fluidx3d/{index}/")
    velocityDir = baseDir / "vtkfiles/velocity/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    V0 = parameters["info"]["conversions"]["V0"]
    L0 = parameters["info"]["conversions"]["L0"]

    velFile = velocityDir / "u_4000000.vtk"
    data = pv.read(velFile)
    Lx, Ly, Lz = data.dimensions
    velocityField = data.get_array("data") * V0
    velocityField = np.reshape(velocityField, (Lx, Ly, Lz, 3), "F")

    rlattice = (np.arange(0, Ly, 1) - Ly / 2 + 0.5) * L0
    rlattice[0] += L0 / 2
    rlattice[-1] -= L0 / 2

    l2error = np.sqrt(
        np.sum((velocityField[1, 1:-1, 1, 0] - u(rlattice[1:-1], VmaxSI)) ** 2)
        / np.sum(u(rlattice[1:-1], VmaxSI) ** 2)
    )
    otherError = np.sqrt(
        np.sum((velocityField[1, 1:-1, 1, 0] - u(rlattice[1:-1], VmaxSI)) ** 2) / (Ly - 2)
    )
    return l2error, otherError / VmaxSI


N = [25, 50, 100, 200]  # It is clear, that these are the exact values
plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
Ns = np.logspace(np.log10(24), np.log10(210), 100)
N2 = (1.0 * np.asarray(Ns)) ** -2
N20 = (1.0 * np.asarray(25)) ** -2
plt.plot(Ns, N2 / N20 * 0.0014831067649990915, "k--")
plt.plot(Ns, N2 / N20 * 0.0008435723395897114, "k--")
plt.plot(Ns, N2 / N20 * 2.09733499e-03, "k--")
i = 0
for fig, R_nu in zip([17, 18], [0.1, 0.7]):
    for Wi in [0.1, 1]:
        _, err = np.genfromtxt(
            f"../data/malaspinas2010/Fig{fig};u;R_ν={R_nu};Wi={Wi}.csv",
            delimiter=";",
            unpack=True,
        )
        plt.plot(N, err, "o", color=color(i), label=f"$R_\\nu={R_nu};Wi={Wi}$")
        e2s = []
        for j in range(4):
            l2e, e2 = process(4 * i + j)
            e2s.append(
                e2
            )  # l2 as defined by me is imho more honest, but malaspinas probably uses the other
        plt.plot(N, e2s, "x", color=color(i), label=f"$R_\\nu={R_nu};Wi={Wi}$")
        i += 1
plt.xscale("log")
plt.yscale("log")
plt.xticks(
    [1e2],
    [r"$10^2$"],
)
plt.xticks(
    [3e1, 4e1, 5e1, 6e1, 7e1, 8e1, 9e1, 2e2],
    [
        r"$3{ }\times{ }10^1$",
        "",
        "",
        r"$6{ }\times{ }10^1$",
        "",
        "",
        "",
        r"$2{ }\times{ }10^2$",
    ],
    minor=True,
)
plt.xlabel("N")
plt.ylabel("L2 error")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.savefig("../plots/malaspinas2010.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
