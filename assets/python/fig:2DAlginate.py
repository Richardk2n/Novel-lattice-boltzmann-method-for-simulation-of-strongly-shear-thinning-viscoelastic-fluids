# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from style import cm
from style import *
from fluidx3d.eval.models import PTT


def process(i):
    baseDir = Path(f"../data/novel_2024-01-15/{i}/")
    cDir = baseDir / "vtkfiles/polymer_conformationTensor/"
    vDir = baseDir / "vtkfiles/velocity/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    V0 = parameters["info"]["conversions"]["V0"]
    L0 = parameters["info"]["conversions"]["L0"]
    T0 = parameters["info"]["conversions"]["T0"]
    rho0 = 1e3
    p0 = rho0 * V0**2
    G = parameters["info"]["px"] * p0 / L0

    cFile = cDir / "CT_2000000000.vtk"
    vFile = vDir / "u_2000000000.vtk"

    data = pv.read(cFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = data.dimensions
    origin = data.origin
    R = (Ly - 2) / 2 * L0

    tauField = data.get_array("data") * PTT.alginate.eta_p / PTT.alginate.lambda_p
    tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    vSlice = vField[Lx // 2, ..., Lz // 2, 0]
    r = (np.arange(0, Ly, 1, dtype=np.float128) + origin[1]) * L0
    r[0] += L0 / 2
    r[-1] -= L0 / 2
    uErr = PTT.alginate.u(np.abs(r))

    rTheory = np.arange(-R, R * (1 + 1e-3), R * 1e-3)
    PTT.alginate.prepareVelocityProfile(R, G, PTT.channel)
    u = PTT.alginate.u(np.abs(rTheory))

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(rTheory * 1e6, u, "k", label="Semi-analytical solution")
    plt.plot(r * 1e6, vSlice, "rx", label="Simulation result")
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$u_\text{x}/\unit{\meter\per\second}$")
    plt.legend()
    plt.savefig("../plots/2DAlginateU.eps")
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(
        r[1:-1] * 1e6,
        vSlice[1:-1] / uErr[1:-1] - 1,
        "rx",
        # label=f"{i}",
    )
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$error$")
    # plt.legend()
    plt.savefig("../plots/2DAlginateErr.eps")
    plt.show()

    err = np.sqrt(np.sum((vSlice[1:-1] - uErr[1:-1]) ** 2) / np.sum(uErr[1:-1] ** 2))

    print(f"{err=}")


if __name__ == "__main__":
    process(0)
