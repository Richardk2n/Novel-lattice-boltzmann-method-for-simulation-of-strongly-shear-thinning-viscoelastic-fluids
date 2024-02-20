# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Sat Oct 14 19:57:01 2023

@author: Richard Kellnberger
"""

import codecs
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import lambertw

from fluidx3d.eval.style import cm, color


def CY(eta0, etaInf, Wi, a1, a2):
    return etaInf + (eta0 - etaInf) / (1 + Wi**a1) ** (a2 / a1)


def readFile(index):
    f = codecs.open(
        f"../data/Daten_SOP_BioPrint_Alginat_PH176/SOP_Bioprint_Probe_{index}.csv", "r", "utf-16"
    )
    lines = f.readlines()
    gdEs = []
    etas = []
    gdNs = []
    Ns = []
    upper = False  # weahter to read the upper intervall (mostly bad)
    if upper:
        for l in lines[402:423]:
            _, _, _, _, _, _, _, _, _, _, eta, gd, _, _, F, _ = l.split("\t")
            gdEs.append(float(gd))
            etas.append(float(eta))
            if float(F) > 0:
                gdNs.append(float(gd))
                Ns.append(float(F))

    for l in lines[428:578]:
        _, _, _, _, _, _, _, _, _, _, eta, gd, _, _, F, _ = l.split("\t")
        gdEs.append(float(gd))
        etas.append(float(eta))
        if float(F) > 0:
            gdNs.append(float(gd))
            Ns.append(float(F))
    idxE = np.argsort(gdEs)
    gdEs = np.asarray(gdEs)[idxE]
    etas = np.asarray(etas)[idxE] / 1e3
    idxN = np.argsort(gdNs)
    gdNs = np.asarray(gdNs)[idxN]
    r = 25 / 2 * 1e-3  # Source: dude trust me bro
    Ns = np.asarray(Ns)[idxN] * 2 / (r**2 * np.pi)
    # plt.plot(gdNs, Ns, "rx")
    d = 0
    # d = 2e-3  # Plate distance; I don't know; This is PP right?
    # The *2 shall signify, that the desired shear rate is present at R/2. I do not know whether this is true
    v = gdNs * d * 2
    rho = 1e3  # aprrox
    # Correction according to https://doi.org/10.1007/BF01525657
    c = 3 / 20 * rho * v**2
    Ns += c
    # plt.plot(gdNs, Ns, "bx")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.ylim(1e-2, 1e3)
    # plt.show()
    # plt.plot(gdEs, etas)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    return gdEs, etas, gdNs, Ns


eta_s = 1e-3


def ptt_eta(gd, eta_p, lam, epsilon):
    return eta_p / np.exp(0.5 * lambertw(4 * epsilon * (gd * lam) ** 2)) + eta_s


def ptt_N(gd, eta_p, lam, epsilon):
    return eta_p / (2 * epsilon * lam) * lambertw(4 * epsilon * (lam * gd) ** 2)


def merge(fun1, x1, y1, fun2, x2, y2):
    log = True
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    if log:
        y = np.log(y)

    def fun(x, *param):
        res = np.concatenate((fun1(x1, *param), fun2(x2, *param)))
        if log:
            return np.log(res)
        else:
            return res

    return x, y, fun


if __name__ == "__main__":
    for i in range(2, 23):
        gde, e, gdn, n = readFile(i)
        plt.plot(gde, e, "x", label=f"{i}")

    plt.xscale("log")
    plt.yscale("log")
    # plt.legend()
    plt.show()

    for i in range(2, 23):
        gde, e, gdn, n = readFile(i)
        plt.plot(gdn, n, "x", label=f"{i}")

    plt.xscale("log")
    plt.yscale("log")
    # plt.legend()
    plt.show()

    gdet = np.asarray([])
    et = np.asarray([])
    gdnt = np.asarray([])
    nt = np.asarray([])
    for i in range(2, 23):
        gde, e, gdn, n = readFile(i)
        gdet = np.concatenate((gdet, gde))
        et = np.concatenate((et, e))
        gdnt = np.concatenate((gdnt, gdn))
        nt = np.concatenate((nt, n))

    x, y, fun = merge(ptt_eta, gdet, et, ptt_N, gdnt, nt)
    params, cov = scipy.optimize.curve_fit(
        fun, x, y, [0.13, 0.0002, 1.56], bounds=([0] * 3, [np.inf] * 3)
    )
    err = np.sqrt(np.diag(cov))
    print(f"fit:")
    print(params)
    print(err)
    gd = np.logspace(np.log10(np.min(gdet)), np.log10(np.max(gdet)) + 5.5, 1000)
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(a)", loc="left")
    plt.plot(gdet, et, "x", label="Data")
    plt.plot(gd, ptt_eta(gd, *params), label="PTT fit")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\dot{\gamma}/\unit{\per\second}$")
    plt.ylabel(r"$\eta/\unit{\pascal\second}$")
    plt.legend()
    plt.savefig("../plots/fit_alginate_eta.pdf", bbox_inches="tight", pad_inches=0)
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(b)", loc="left")
    plt.plot(gdnt, nt, "x", label="Data")
    plt.plot(gd, ptt_N(gd, *params), label="PTT fit")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\dot{\gamma}/\unit{\per\second}$")
    plt.ylabel(r"$N_1/\unit{\pascal}$")
    plt.legend()
    plt.savefig("../plots/fit_alginate_N.pdf", bbox_inches="tight", pad_inches=0)
