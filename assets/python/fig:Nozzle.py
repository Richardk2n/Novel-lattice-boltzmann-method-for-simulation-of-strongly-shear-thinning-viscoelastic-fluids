# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pyvista as pv
import scipy
from scipy.special import lambertw
from style import *

eta_s = 1e-3


def process():
    baseDir = Path("../data/Nozzle/2/")
    cDir = baseDir / "vtkfiles/polymerConformationTensor/"
    sDir = baseDir / "vtkfiles/strainRateTensor/"
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
    # print("###")
    # print(parameters["info"]["px"])
    # print(p0 / L0)
    # print(G)

    # cFile = cDir / "C_500000.vtk"
    # sFile = sDir / "S_500000.vtk"
    vFile = vDir / "u_500000.vtk"
    eta_p_SI = 18.7e-3
    lambda_SI = 0.344e-3
    epsilon = 0.27
    # print(uHP(0, G, eta_p_SI))
    # print("---")
    # data = pv.read(cFile)
    # dataS = pv.read(sFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = dataV.dimensions
    # tauField = data.get_array("data") * eta_p_SI / lambda_SI
    # tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    # sField = dataS.get_array("data") / T0
    # sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    if False:
        mask = np.zeros(vField.shape)

        for x in range(Lx):
            localR = (10.5 - 94.5) / (Lx - 3) * (x - 1) + 94.5
            for y in range(Ly):
                for z in range(Lz):
                    if np.hypot(y - (Ly / 2.0) + 0.5, z - (Lz / 2.0) + 0.5) >= localR + 0.5:
                        for i in range(3):
                            mask[x, y, z, i] = 1

        vField = ma.masked_array(vField, mask=mask)

    vSlice = vField[..., Lz // 2, 0]
    # tau12 = tauField[..., Lz // 2, 3]
    # D12 = sField[..., Lz // 2, 3]
    # eta = tau12 / D12 / 2 + 1e-3

    inflow = np.sum(vField[0, ..., 0])
    outflow = np.sum(vField[Lx - 1, ..., 0])
    print(inflow)
    print(outflow)

    x = np.arange(0, Lx) - 0.5  # Offset?
    localR = (10.5 - 94.5) / (Lx - 3) * (x - 1) + 94.5
    # lower = 94.58898305084746 - localR
    # upper = 94.58898305084746 + localR
    upper = 95 + localR
    # print(upper)
    lower = 95 - localR + 1  # Something is wrong here
    # print(lower)

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    # plt.plot(x * L0 * 1e3, upper * L0 * 1e3, "k")
    # plt.plot(x * L0 * 1e3, lower * L0 * 1e3, "k")
    plt.imshow(
        vSlice.T,
        cmap="coolwarm",
        extent=[
            -0.5 * L0 * 1e3,
            (Lx - 1 + 0.5) * L0 * 1e3,
            -0.5 * L0 * 1e3,
            (Ly - 1 + 0.5) * L0 * 1e3,
        ],
    )
    plt.xlabel(r"$x/\unit{\milli\meter}$")
    plt.ylabel(r"$y/\unit{\milli\meter}$")
    plt.colorbar(location="top", label=r"$v_x/\unit{\meter\per\second}$")
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    plt.savefig("../plots/Nozzle_v.eps")
    plt.show()

    r"""
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(tau12.T, cmap="coolwarm", extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6])
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$\tau_{12}/\unit{\pascal}$")
    # plt.savefig(f"../plots/Nozzle_tau.eps")
    plt.show()
    """

    r"""
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(D12.T, cmap="coolwarm", extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6])
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$D_{12}/\unit{\per\second}$")
    # plt.savefig(f"../plots/Nozzle.eps")
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(
        eta.T * 1e3,
        cmap="coolwarm",
        extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6],
        vmin=0,
        vmax=40,
    )
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$\eta/\unit{\pascal\second}$")
    # plt.savefig(f"../plots/Nozzle.eps")
    plt.show()
    """


if __name__ == "__main__":
    t1 = time()
    process()
    print(time() - t1)
