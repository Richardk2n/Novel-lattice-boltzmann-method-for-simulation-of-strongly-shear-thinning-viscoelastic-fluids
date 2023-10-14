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
from scipy.special import lambertw
from style import *

eta_s = 1e-3


def ptt_eta(gd, eta_p, lam, epsilon):
    return eta_p / np.exp(0.5 * lambertw(4 * epsilon * (gd * lam) ** 2)) + eta_s


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
    eta_p_SI = 18.7e-3
    lambda_SI = 0.344e-3
    data = pv.read(cFile)
    dataS = pv.read(sFile)
    Lx, Ly, Lz = data.dimensions
    tauField = data.get_array("data") * eta_p_SI / lambda_SI
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
plt.plot(gds * 0.344e-3, ptt_eta(gds, 18.7e-3, 0.344e-3, 0.27) * 1e3, "k", label="Theory")
plt.plot(gd * 0.344e-3, etas, "rx", label="Simulation")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$Wi$")
plt.ylabel(r"$\eta/\unit{\milli\pascal}$")
plt.legend()
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.ylim(19.699998804, 19.699998806)
plt.savefig("../plots/wiParameterStudy.eps")
plt.show()
plt.plot(
    gd,
    np.abs(ptt_eta(gd, 18.7e-3, 0.344e-3, 0.27) * 1e3 - etas)
    / (ptt_eta(gd, 18.7e-3, 0.344e-3, 0.27) * 1e3),
    "rx",
)
plt.xscale("log")
plt.yscale("log")
