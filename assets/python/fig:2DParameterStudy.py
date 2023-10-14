# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
import sys
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy
from scipy.special import lambertw
from multiprocessing import Pool
from style import *

eta_s = 1e-3
j = 0
R = 10e-6


def ptt_eta(gd, eta_p, lam, epsilon):
    return eta_p / np.exp(0.5 * lambertw(4 * epsilon * (gd * lam) ** 2)) + eta_s


def getl2(G, eta_p, lam, epsilon):
    return np.float128(2 ** (2 * j - 1) * eta_p**2 / (epsilon * lam**2 * G**2))


# According to Oliveira 1999
def uS(r, G, eta_p, lam, epsilon):  # eta_s = 0
    l2 = getl2(G, eta_p, lam, epsilon)
    return G * l2 / (2 ** (j + 1) * (eta_p + eta_s)) * (np.exp(R**2 / l2) - np.exp(r**2 / l2))


def uHP(r, G, eta_p):
    return G / (2 * (eta_p + eta_s)) * (R**2 - r**2)


def getFun(r, G, lam, eta_p, epsilon):
    def fun(x):
        return (
            2 * epsilon * lam**2 / eta_p**2 * (-G * r / 2**j - eta_s * x) ** 2
            - np.log(eta_p)
            + np.log(-G * r / 2**j / x - eta_s)
        )

    return fun


def calc(r, G, lam, eta_p, epsilon):
    guess = -G * r / 2**j / eta_s / 10
    fun = getFun(r, G, lam, eta_p, epsilon)
    answ = scipy.optimize.root(fun, guess)
    return answ.x[0]


def findRoot(G, lam, eta_p, epsilon):
    dr = R * 1e-5
    r = np.arange(0, R, dr)
    ru = np.arange(dr, R + dr, dr)

    param = [(re, G, lam, eta_p, epsilon) for re in r[1:]]

    with Pool(32) as p:
        res = p.starmap(calc, param)

    gds = np.concatenate(([0], res))

    us = []
    for i in range(len(r)):
        us.append(np.sum(gds[:i]) * dr)
    us = np.asarray(us) - us[-1]

    return ru, us


def approximate(r, r_ref, u_ref):  # poor approximation
    res = []
    for re in r:
        i = np.argmin(np.abs(r_ref - re))
        indices = np.argpartition(np.abs(re - r_ref), (0, 1))
        i1 = np.min((indices[0], indices[1]))
        i2 = np.max((indices[0], indices[1]))

        m = (u_ref[i2] - u_ref[i1]) / (r_ref[i2] - r_ref[i1])

        res.append(m * (re - r_ref[i1]) + u_ref[i1])

    return np.asarray(res)


def process(index):
    baseDir = Path(f"../data/2DParameterStudy/{index}/")
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

    tau12 = tauField[Lx // 2, ..., Lz // 2, 3]
    D12 = sField[Lx // 2, ..., Lz // 2, 3]
    # shearRateSI = parameters["shearRateSI"]
    eta = tau12 / D12 / 2 + 1e-3

    vSlice = vField[Lx // 2, ..., Lz // 2, 0]
    r = (np.arange(0, Ly, 1, dtype=np.float128) - Ly / 2 + 0.5) * L0
    r[0] += L0 / 2
    r[-1] -= L0 / 2

    rsa, usa = findRoot(G, lambda_SI, eta_p_SI, epsilon)

    rsa = np.concatenate((-np.flip(rsa[1:]), rsa))
    usa = np.concatenate((np.flip(usa[1:]), usa))

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(rsa[::10] * 1e6, usa[::10], "k", label=f"Theory {i}")
    # plt.plot(r * 1e6, uS(r, G, eta_p_SI, lambda_SI, epsilon), "g", label=f"Theory {i}")
    plt.plot(r * 1e6, vSlice, "rx", label=f"Data {i}")
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$u_\text{x}/\unit{\meter\per\second}$")
    plt.legend()
    plt.savefig(f"../plots/2DParameterStudyU{i}.eps")
    plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(
        r[1:-1] * 1e6,
        vSlice[1:-1] / approximate(r[1:-1], rsa, usa) - 1,
        "rx",
        label=f"{i}",
    )
    plt.xlabel(r"$r/\unit{\micro\meter}$")
    plt.ylabel(r"$error$")
    plt.legend()
    plt.savefig(f"../plots/2DParameterStudyErr{i}.eps")
    plt.show()

    err = np.sqrt(
        np.sum((vSlice[1:-1] - approximate(r[1:-1], rsa, usa)) ** 2)
        / np.sum(approximate(r[1:-1], rsa, usa) ** 2)
    )

    return eta, D12, G, err


if __name__ == "__main__":
    t1 = time()

    etas = []
    ds = []
    Gs = []
    errs = []
    for i in range(15):
        eta, D12, G, err = process(i)
        etas.append(eta)
        ds.append(D12)
        Gs.append(G)
        errs.append(err)

    gd = np.logspace(0, 7, 15)
    gds = np.logspace(0, 7, 151)
    etas = np.asarray(etas) * 1e3

    print(time() - t1)

    for i in range(15):
        plt.plot(etas[i], "rx", label=f"{i}")
        plt.legend()
        plt.show()

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.plot(Gs, errs, "rx")
    plt.xscale("log")
    plt.xlabel(r"$-\frac{\partial p}{\partial x}/\unit{\pascal\per\meter}$")
    plt.ylabel(r"$error$")
    plt.savefig("../plots/2DParameterStudyErr.eps")
    plt.show()
