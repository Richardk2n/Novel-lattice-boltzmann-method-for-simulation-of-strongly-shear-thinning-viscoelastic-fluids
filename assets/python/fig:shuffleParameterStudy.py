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
from scipy.special import lambertw

from fluidx3d.eval.models import PTT
from fluidx3d.eval.style import cm


def process(index):
    baseDir = Path(f"../data/SuffleParameterStudy/{index}/")
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

    cFile = cDir / "C_10000000.vtk"
    sFile = sDir / "S_10000000.vtk"
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
for i in range(11):
    eta, D12 = process(i)
    etas.append(eta)
    ds.append(D12)

alpha_s = np.arange(1, 2.1, 0.1)
etas = np.asarray(etas) * 1e3

plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
plt.title("(b)", loc="left")
plt.plot(alpha_s, etas, "rx")
# plt.plot(alpha_s, [PTT.mc0_49.eta(1) * 1e3] * len(alpha_s))
plt.xlabel(r"$\alpha_\text{s}$")
plt.ylabel(r"$\eta/\unit{\milli\pascal\second}$")
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.ylim(19.699998804, 19.699998806)
plt.savefig("../plots/suffleParameterStudy.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
