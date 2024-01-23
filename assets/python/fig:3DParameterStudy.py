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


def process(index):
    baseDir = Path(f"../data/3DParameterStudy/{index}/")
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

    cFile = cDir / "C_5000000.vtk"
    sFile = sDir / "S_5000000.vtk"
    vFile = vDir / "u_5000000.vtk"
    # print(uHP(0, G, eta_p_SI))
    # print("---")
    data = pv.read(cFile)
    dataS = pv.read(sFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = data.dimensions
    # origin = data.origin
    R = (Ly - 2) / 2 * L0

    tauField = data.get_array("data") * PTT.mc0_49.eta_p / PTT.mc0_49.lambda_p
    tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    sField = dataS.get_array("data") / T0
    sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    tau12 = tauField[Lx // 2, ..., Lz // 2, 3]
    D12 = sField[Lx // 2, ..., Lz // 2, 3]
    # shearRateSI = parameters["shearRateSI"]
    eta = tau12 / D12 / 2 + PTT.mc0_49.eta_s

    vSlice = vField[Lx // 2, ..., Lz // 2, 0]
    r = (np.arange(0, Ly, 1, dtype=np.float128) - Ly / 2 + 0.5) * L0
    r[0] += L0 / 2
    r[-1] -= L0 / 2
    PTT.mc0_49.prepareVelocityProfile(R, G, PTT.pipe)
    uErr = PTT.mc0_49.u(np.abs(r))

    rTheory = np.arange(-R, R * (1 + 1e-3), R * 1e-3)
    u = PTT.mc0_49.u(np.abs(rTheory))

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(rTheory * 1e6, u, "k", label="Semi-analytical solution")
    plt.plot(r * 1e6, vSlice, "rx", label=f"Data {i}")
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$u_\text{x}/\unit{\meter\per\second}$")
    plt.legend()
    plt.savefig(f"../plots/3DParameterStudyU{i}.eps")
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(
        r[1:-1] * 1e6,
        vSlice[1:-1] / uErr[1:-1] - 1,
        "rx",
        label=f"{i}",
    )
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$error$")
    plt.legend()
    plt.savefig(f"../plots/3DParameterStudyErr{i}.eps")
    plt.show()

    err = np.sqrt(np.sum((vSlice[1:-1] - uErr[1:-1]) ** 2) / np.sum(uErr[1:-1] ** 2))

    return eta, D12, G, err


def plotWorst(index=7):
    baseDir = Path(f"../data/3DParameterStudy/{index}/")
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

    cFile = cDir / "C_5000000.vtk"
    sFile = sDir / "S_5000000.vtk"
    vFile = vDir / "u_5000000.vtk"
    # print(uHP(0, G, eta_p_SI))
    # print("---")
    data = pv.read(cFile)
    dataS = pv.read(sFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = data.dimensions
    # origin = data.origin
    R = (Ly - 2) / 2 * L0

    tauField = data.get_array("data") * PTT.mc0_49.eta_p / PTT.mc0_49.lambda_p
    tauField = np.reshape(tauField, (Lx, Ly, Lz, 6), "F")
    sField = dataS.get_array("data") / T0
    sField = np.reshape(sField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    tau12 = tauField[Lx // 2, ..., Lz // 2, 3]
    D12 = sField[Lx // 2, ..., Lz // 2, 3]
    # shearRateSI = parameters["shearRateSI"]
    eta = tau12 / D12 / 2 + PTT.mc0_49.eta_s

    vSlice = vField[Lx // 2, ..., Lz // 2, 0]
    r = (np.arange(0, Ly, 1, dtype=np.float128) - Ly / 2 + 0.5) * L0
    r[0] += L0 / 2
    r[-1] -= L0 / 2
    PTT.mc0_49.prepareVelocityProfile(R, G, PTT.pipe)
    uErr = PTT.mc0_49.u(np.abs(r))

    rTheory = np.arange(-R, R * (1 + 1e-3), R * 1e-3)
    u = PTT.mc0_49.u(np.abs(rTheory))

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(rTheory * 1e6, u, "k", label="Semi-analytical solution")
    plt.plot(r * 1e6, vSlice, "rx", label="Simulation result")
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$u_\text{x}/\unit{\meter\per\second}$")
    plt.legend()
    plt.savefig("../plots/3DParameterStudyUWorst.eps")
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
    plt.savefig("../plots/3DParameterStudyErrWorst.eps")
    plt.show()

    err = np.sqrt(np.sum((vSlice[1:-1] - uErr[1:-1]) ** 2) / np.sum(uErr[1:-1] ** 2))

    return eta, D12, G, err


if __name__ == "__main__":
    t1 = time()

    etas = []
    ds = []
    Gs = []
    errs = []
    for i in range(12):  # 12 13 14 are failures
        eta, D12, G, err = process(i)
        etas.append(eta)
        ds.append(D12)
        Gs.append(G)
        errs.append(err)

    gd = np.logspace(0, 7, 15)
    gds = np.logspace(0, 7, 151)
    etas = np.asarray(etas) * 1e3

    print(time() - t1)

    for i in range(12):  # 12 13 14 are failures
        plt.plot(etas[i], "rx", label=f"{i}")
        plt.legend()
        plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(Gs, errs, "rx")
    plt.xscale("log")
    plt.xlabel(r"$-\frac{\partial p}{\partial x}/\unit{\pascal\per\meter}$")
    plt.ylabel(r"$error$")
    plt.savefig("../plots/3DParameterStudyErr.eps")
    plt.show()

    plotWorst()
