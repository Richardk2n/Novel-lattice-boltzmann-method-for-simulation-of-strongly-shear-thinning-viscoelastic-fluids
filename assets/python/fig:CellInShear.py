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

from fluidx3d.eval.models import PTT
from fluidx3d.eval.style import cm


def process():
    baseDir = Path("../data/cellInShear_2024-01-23/0/")
    cDir = baseDir / "vtkfiles/polymer_conformationTensor/"
    sDir = baseDir / "vtkfiles/strainRateTensor/"
    vDir = baseDir / "vtkfiles/velocity/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    # V0 = parameters["info"]["conversions"]["V0"]
    L0 = 1e-6  # parameters["info"]["conversions"]["L0"]
    mu_SI = 1e-3 + 2 * PTT.mc0_49.eta_p
    nu_SI = mu_SI / 1e3
    nu0 = 6 * nu_SI
    T0 = L0**2 / nu0  # parameters["info"]["conversions"]["T0"]
    V0 = L0 / T0
    rho0 = 1e3
    p0 = rho0 * V0**2
    # print("###")
    # print(parameters["info"]["px"])
    # print(p0 / L0)
    # print(G)

    cFile = cDir / "CT_40000000.vtk"
    # sFile = sDir / "S_500000.vtk"
    vFile = vDir / "u_40000000.vtk"

    data = pv.read(cFile)
    # dataS = pv.read(sFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = dataV.dimensions
    tauField = data.get_array("data") * PTT.mc0_49.eta_p / PTT.mc0_49.lambda_p
    tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    # sField = dataS.get_array("data") / T0
    # sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    vSlice = vField[..., 1:-1, Lz // 2, 0]

    deviator = [[0] * 3] * 3
    idx = [[0, 3, 5], [3, 1, 4], [5, 4, 2]]

    vonMieses = np.zeros((Lx, Ly, Lz))
    for i in range(3):
        for j in range(3):
            deviator[i][j] = tauField[..., idx[i][j]] - (
                np.sum(tauField[..., :3], axis=-1) / 3 if idx[i][j] < 3 else 0  # Assume correct?
            )

            vonMieses += 3 / 2 * deviator[i][j] * deviator[i][j]
    vonMieses = np.sqrt(vonMieses)
    vonMiesesSlice = vonMieses[..., 1:-1, Lz // 2]

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))

    plt.imshow(
        vSlice.T,
        cmap="coolwarm",
        extent=[
            -0.5 * L0 * 1e3,
            (Lx - 1 + 0.5) * L0 * 1e3,
            0.5 * L0 * 1e3,
            (Ly - 1 - 0.5) * L0 * 1e3,
        ],
    )
    plt.xlabel(r"$x/\unit{\milli\meter}$")
    plt.ylabel(r"$y/\unit{\milli\meter}$")
    plt.colorbar(location="right", label=r"$v_x/\unit{\meter\per\second}$")
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["right"].set_visible(False)
    plt.savefig("../plots/cellInShear_v.pdf", dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(b)", loc="left")
    plt.imshow(
        vonMiesesSlice.T,
        cmap="coolwarm",
        extent=[
            -0.5 * L0 * 1e3,
            (Lx - 1 + 0.5) * L0 * 1e3,
            0.5 * L0 * 1e3,
            (Ly - 1 - 0.5) * L0 * 1e3,
        ],
        # interpolation="none",
    )
    plt.xlabel(r"$x/\unit{\milli\meter}$")
    plt.ylabel(r"$y/\unit{\milli\meter}$")
    plt.colorbar(location="right", label=r"$\sigma_\text{vM}/\unit{\pascal}$")
    plt.savefig("../plots/cellInShear_sigma_vM.pdf", dpi=1200, bbox_inches="tight", pad_inches=0)
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
