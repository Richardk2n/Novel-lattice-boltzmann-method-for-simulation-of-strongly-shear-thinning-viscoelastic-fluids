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
    gdEs, etas = np.genfromtxt(
        f"../data/methylcellulose/cc_cp_visc_0_{index}.txt", delimiter=";", unpack=True
    )
    gdNs, Ns = np.genfromtxt(
        f"../data/methylcellulose/cp_normal_0_{index}.txt", delimiter=";", unpack=True
    )
    etas /= 1e3

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
    for i in [49, 59, 83]:
        gde, e, gdn, n = readFile(i)
        plt.plot(gde, e, "x", label=f"{i}")

    plt.xscale("log")
    plt.yscale("log")
    # plt.legend()
    plt.show()

    for i in [49, 59, 83]:
        gde, e, gdn, n = readFile(i)
        plt.plot(gdn, n, "x", label=f"{i}")

    plt.xscale("log")
    plt.yscale("log")
    # plt.legend()
    plt.show()

    def f(i):
        if i == 49:
            return 0
        if i == 59:
            return 1
        if i == 83:
            return 2

    gd = np.logspace(np.log10(np.min(gde)), np.log10(np.max(gde)) + 4, 1000)
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(a)", loc="left")
    for i in [49, 59, 83]:
        gde, e, gdn, n = readFile(i)
        plt.plot(gde, e, "x", color=color(f(i)), label=f"Data $\\SI{{0.{i}}}{{\\percent}}$")

        x, y, fun = merge(ptt_eta, gde, e, ptt_N, gdn, n)
        params, cov = scipy.optimize.curve_fit(
            fun, x, y, [0.20, 0.001, 0.3], bounds=([0] * 3, [np.inf] * 3)
        )
        err = np.sqrt(np.diag(cov))
        print(f"fit {i}:")
        print(params)
        print(err)
        plt.plot(
            gd,
            ptt_eta(gd, *params),
            color=color(f(i)),
            label=f"PTT fit $\\SI{{0.{i}}}{{\\percent}}$",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\dot{\gamma}/\unit{\per\second}$")
    plt.ylabel(r"$\eta/\unit{\pascal\second}$")
    plt.legend()
    plt.savefig("../plots/fit_mc_eta.pdf", bbox_inches="tight", pad_inches=0)
    gd = np.logspace(np.log10(np.min(gdn)), np.log10(np.max(gdn)) + 4, 1000)
    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(b)", loc="left")
    for i in [49, 59, 83]:
        gde, e, gdn, n = readFile(i)
        x, y, fun = merge(ptt_eta, gde, e, ptt_N, gdn, n)
        params, cov = scipy.optimize.curve_fit(
            fun, x, y, [0.20, 0.001, 0.3], bounds=([0] * 3, [np.inf] * 3)
        )
        plt.plot(gdn, n, "x", color=color(f(i)), label=f"Data $\\SI{{0.{i}}}{{\\percent}}$")
        plt.plot(
            gd, ptt_N(gd, *params), color=color(f(i)), label=f"PTT fit $\\SI{{0.{i}}}{{\\percent}}$"
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\dot{\gamma}/\unit{\per\second}$")
    plt.ylabel(r"$N_1/\unit{\pascal}$")
    plt.legend()
    plt.savefig("../plots/fit_mc_N.pdf", bbox_inches="tight", pad_inches=0)
