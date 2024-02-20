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

from fluidx3d.eval.models import PTT
from fluidx3d.eval.style import cm


def process(index):
    baseDir = Path(f"../data/wiParameterStudy/{index}/")
    cDir = baseDir / "vtkfiles/polymerConformationTensor/"
    sDir = baseDir / "vtkfiles/strainRateTensor/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    V0 = parameters["info"]["conversions"]["V0"]
    L0 = parameters["info"]["conversions"]["L0"]
    T0 = parameters["info"]["conversions"]["T0"]
    rho0 = 1e3

    cFile = cDir / "C_5000000.vtk"
    sFile = sDir / "S_5000000.vtk"
    data = pv.read(cFile)
    dataS = pv.read(sFile)
    Lx, Ly, Lz = data.dimensions
    tauField = data.get_array("data") * PTT.mc0_49.eta_p / PTT.mc0_49.lambda_p
    tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    sField = dataS.get_array("data") / T0
    sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")

    tau12 = tauField[Lx // 2, Ly // 2, Lz // 2, 3]
    D12 = sField[Lx // 2, Ly // 2, Lz // 2, 3]
    shearRateSI = parameters["shearRateSI"]
    eta = tau12 / shearRateSI + 1e-3
    return eta, D12


etas = []
ds = []
for i in range(15):
    eta, D12 = process(i)
    etas.append(eta)
    ds.append(D12)

gd = np.logspace(0, 7, 15)
gds = np.logspace(0, 7, 151)
etas = np.asarray(etas) * 1e3

plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
plt.title("(a)", loc="left")
plt.plot(gds, PTT.mc0_49.eta(gds) * 1e3, "k", label="Theory")
plt.plot(gd, etas, "rx", label="Simulation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\dot{\gamma}/\unit{\per\second}$")
plt.ylabel(r"$\eta/\unit{\milli\pascal\second}$")
plt.legend()
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.ylim(19.699998804, 19.699998806)
plt.savefig("../plots/wiParameterStudy.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
plt.plot(
    gd,
    np.abs(PTT.mc0_49.eta(gd) * 1e3 - etas) / (PTT.mc0_49.eta(gd) * 1e3),
    "rx",
)
plt.xscale("log")
plt.yscale("log")
