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
import numpy.ma as ma
import pyvista as pv

from fluidx3d.eval.models import PTT
from fluidx3d.eval.style import cm


def process():
    baseDir = Path("../data/RT-DC/0/")
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
    G = parameters["info"]["px"] * p0 / L0
    # print("###")
    # print(parameters["info"]["px"])
    # print(p0 / L0)
    # print(G)

    cFile = cDir / "C_500000.vtk"
    sFile = sDir / "S_500000.vtk"
    vFile = vDir / "u_500000.vtk"
    eta_p_SI = 18.7e-3
    lambda_SI = 0.344e-3
    epsilon = 0.27
    # print(uHP(0, G, eta_p_SI))
    # print("---")
    data = pv.read(cFile)
    dataS = pv.read(sFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = data.dimensions
    tauField = data.get_array("data") * eta_p_SI / lambda_SI
    tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    sField = dataS.get_array("data") / T0
    sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

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

    if True:
        maskvM = np.zeros(vonMieses.shape)
        mask = np.zeros(vField.shape)
        mask2 = np.zeros(tauField.shape)
        width = 30
        length = 225
        padding = 60

        for x in range(Lx):
            for y in range(Ly):
                for z in range(Lz):
                    if x < padding:
                        if y == 0 or y == Ly - 1 or z == 0 or z == Lz - 1:
                            maskvM[x, y, z] = 1
                            for i in range(3):
                                mask[x, y, z, i] = 1
                                mask2[x, y, z, 2 * i] = 1
                                mask2[x, y, z, 2 * i + 1] = 1
                    elif x < padding + width:
                        if (
                            y < x - (padding - 2)
                            or y >= Ly - x + (padding - 2)
                            or z == 0
                            or z == Lz - 1
                        ):
                            maskvM[x, y, z] = 1
                            for i in range(3):
                                mask[x, y, z, i] = 1
                                mask2[x, y, z, 2 * i] = 1
                                mask2[x, y, z, 2 * i + 1] = 1
                    elif x < padding + width + length:
                        if y < width + 1 or y >= 2 * width + 1 or z == 0 or z == Lz - 1:
                            maskvM[x, y, z] = 1
                            for i in range(3):
                                mask[x, y, z, i] = 1
                                mask2[x, y, z, 2 * i] = 1
                                mask2[x, y, z, 2 * i + 1] = 1
                    elif x < padding + 2 * width + length:
                        if (
                            y < (padding + width + length) - x + (width + 1)
                            or y >= x - (padding + width + length) + (2 * width + 1)
                            or z == 0
                            or z == Lz - 1
                        ):
                            maskvM[x, y, z] = 1
                            for i in range(3):
                                mask[x, y, z, i] = 1
                                mask2[x, y, z, 2 * i] = 1
                                mask2[x, y, z, 2 * i + 1] = 1
                    else:
                        if y == 0 or y == Ly - 1 or z == 0 or z == Lz - 1:
                            maskvM[x, y, z] = 1
                            for i in range(3):
                                mask[x, y, z, i] = 1
                                mask2[x, y, z, 2 * i] = 1
                                mask2[x, y, z, 2 * i + 1] = 1

        vonMieses = ma.masked_array(vonMieses, mask=maskvM)
        vField = ma.masked_array(vField, mask=mask)
        tauField = ma.masked_array(tauField, mask=mask2)

    vonMiesesSlice = vonMieses[..., Lz // 2]

    vSlice = vField[..., Lz // 2, 0]
    tau12 = tauField[..., Lz // 2, 3]
    D12 = sField[..., Lz // 2, 3]
    eta = tau12 / D12 / 2 + PTT.mc0_49.eta_s

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(a)", loc="left", pad=45, x=-0.11)
    plt.imshow(
        vSlice.T,
        cmap="coolwarm",
        extent=[0, (Lx - 1) * L0 * 1e6, -Ly / 2 * L0 * 1e6, Ly / 2 * L0 * 1e6],
    )
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$v_x/\unit{\meter\per\second}$")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.savefig("../plots/RT-DC_v.pdf", dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(b)", loc="left", pad=45, x=-0.11)
    plt.imshow(
        vonMiesesSlice.T,
        cmap="coolwarm",
        extent=[0, (Lx - 1) * L0 * 1e6, -Ly / 2 * L0 * 1e6, Ly / 2 * L0 * 1e6],
    )
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$\sigma_\text{vM}/\unit{\pascal}$")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.savefig("../plots/RT-DC_sigma_vM.pdf", dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(c)", loc="left", pad=45, x=-0.11)
    plt.imshow(
        tau12.T,
        cmap="coolwarm",
        extent=[0, (Lx - 1) * L0 * 1e6, -Ly / 2 * L0 * 1e6, Ly / 2 * L0 * 1e6],
    )
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$\tau_{12}/\unit{\pascal}$")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.savefig("../plots/RT-DC_tau.pdf", dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.show()

    r"""
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.imshow(D12.T, cmap="coolwarm", extent=[0, (Lx - 1) * L0 * 1e6, 0, (Ly - 1) * L0 * 1e6])
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$D_{12}/\unit{\per\second}$")
    # plt.savefig(f"../plots/RT-DC.eps")
    plt.show()
    """

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(b)", loc="left", pad=45, x=-0.1)
    plt.imshow(
        eta.T * 1e3,
        cmap="coolwarm",
        extent=[0, (Lx - 1) * L0 * 1e6, -Ly / 2 * L0 * 1e6, Ly / 2 * L0 * 1e6],
        vmin=-100,
        vmax=100,
    )
    plt.xlabel(r"$x/\unit{\micro\meter}$")
    plt.ylabel(r"$y/\unit{\micro\meter}$")
    plt.colorbar(location="top", label=r"$\eta/\unit{\milli\pascal\second}$")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.savefig("../plots/RT-DC_eta.pdf", dpi=1200, bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    t1 = time()
    process()
    print(time() - t1)
