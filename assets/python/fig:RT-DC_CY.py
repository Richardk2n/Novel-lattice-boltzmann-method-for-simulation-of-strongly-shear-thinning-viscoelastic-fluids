# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from fluidx3d.eval.models.CY import mc0_49
from fluidx3d.eval.style import cm


def process():
    baseDir = Path("../data/RT-DC/1/")
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
    G = parameters["info"]["px"] * p0 / L0
    # print("###")
    # print(parameters["info"]["px"])
    # print(p0 / L0)
    # print(G)

    sFile = sDir / "S_500000.vtk"
    vFile = vDir / "u_500000.vtk"
    # print(uHP(0, G, eta_p_SI))
    # print("---")
    dataS = pv.read(sFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = dataS.dimensions
    sField = dataS.get_array("data") / T0
    sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    vSlice = vField[..., Lz // 2, 0]
    D12 = sField[..., Lz // 2, 3]
    eta = mc0_49.eta(np.abs(D12))
    tau12 = 2 * eta * D12

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(vSlice.T, cmap="coolwarm", extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6])
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$v_x/\unit{\meter\per\second}$")
    plt.savefig(f"../plots/RT-DC_CY_v.eps")
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(tau12.T, cmap="coolwarm", extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6])
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$\tau_{12}/\unit{\pascal}$")
    plt.savefig(f"../plots/RT-DC_CY_tau.eps")
    plt.show()

    r"""
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(D12.T, cmap="coolwarm", extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6])
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$D_{12}/\unit{\per\second}$")
    # plt.savefig(f"../plots/RT-DC.eps")
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
    # plt.savefig(f"../plots/RT-DC.eps")
    plt.show()
    """


if __name__ == "__main__":
    t1 = time()
    process()
    print(time() - t1)
