# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Jan 25 15:23:55 2024

@author: Richard Kellnberger
"""

from typing import Annotated, Tuple, Optional
from functools import wraps

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from multiprocessing import Pool
import warnings


def pooled(fun):
    @wraps(fun)
    def decorated(*args):
        with Pool() as p:
            res = np.asarray(p.starmap(fun, np.transpose(args)))
        return res

    return decorated


def eq18(alpha1s: float, alpha2s: float, alpha3s: float) -> Tuple[float, float, float]:
    def integrand(lambda_: float, alphas: float) -> float:
        Deltas: float = (alpha1s + lambda_) * (alpha2s + lambda_) * (alpha3s + lambda_)
        return lambda_ * (alphas + lambda_) / Deltas**1.5

    g1pp, _ = quad(integrand, 0, np.inf, (alpha1s,))
    g2pp, _ = quad(integrand, 0, np.inf, (alpha2s,))
    g3pp, _ = quad(integrand, 0, np.inf, (alpha3s,))

    return g1pp, g2pp, g3pp


def eq18r(alpha1s: float, alpha2s: float, alpha3s: float) -> Tuple[float, float]:
    """
    Like eq18 but only calculates g_1'' and g_2''.

    Parameters
    ----------
    alpha1s : float
            &alpha;_1 squared.
    alpha2s : float
            &alpha;_2 squared.
    alpha3s : float
            &alpha;_3 squared.

    Returns
    -------
    Tuple[float, float]
            g_1'' and g_2''.

    """

    def integrand(lambda_: float, alphas: float) -> float:
        Deltas: float = (alpha1s + lambda_) * (alpha2s + lambda_) * (alpha3s + lambda_)
        return lambda_ * (alphas + lambda_) / Deltas**1.5

    g1pp, _ = quad(integrand, 0, np.inf, (alpha1s,))
    g2pp, _ = quad(integrand, 0, np.inf, (alpha2s,))

    return g1pp, g2pp


def eq21(alpha1s: float, alpha2s: float, alpha3s: float) -> float:
    def integrand(lambda_: float, alphas: float) -> float:
        Deltas: float = (alpha1s + lambda_) * (alpha2s + lambda_) * (alpha3s + lambda_)
        return (alphas + lambda_) / Deltas**1.5

    g3p, _ = quad(integrand, 0, np.inf, (alpha3s,))

    return g3p


def eq39(alpha1s: float, alpha2s: float, alpha3s: float) -> float:
    g1pp, g2pp, g3pp = np.transpose(pooled(eq18)(alpha1s, alpha2s, alpha3s))
    denominator: float = g2pp * g3pp + g3pp * g1pp + g1pp * g2pp
    I: float = 0.4 * (g1pp + g2pp) / denominator

    return I


def eq39_40(alpha1s: float, alpha2s: float, alpha3s: float) -> float:
    g1pp, g2pp = eq18r(alpha1s, alpha2s, alpha3s)

    return (g1pp - g2pp) / (g1pp + g2pp)


def eq43(alpha1s: float, alpha2s: float, alpha3s: float) -> Annotated[float, "K"]:
    return (alpha1s + alpha2s) / (5 * pooled(eq21)(alpha1s, alpha2s, alpha3s) * alpha1s * alpha2s)


def solve_eq78(alpha1s: float) -> Annotated[float, "alpha2s"]:
    def eq78(alpha2s: float) -> float:
        alpha3s: float = 1 / (alpha1s * alpha2s)
        JbyI = eq39_40(alpha1s, alpha2s, alpha3s)

        return JbyI * (alpha1s - alpha2s) - (alpha1s + alpha2s - 2 * alpha3s)

    sol = root_scalar(
        eq78,
        method="bisect",
        bracket=[1e-15, 1 / alpha1s],
    )

    if not sol.converged:
        print(f"{sol=}")
        raise Exception("Could not find alpha2")

    return sol.root


def eq79(
    alpha1s: float, alpha2s: float, alpha3s: float, I: float, theta2: float, sigma: float
) -> Annotated[float, "kappa"]:
    # typo in Roscoe! # TODO CHECK
    numerator = alpha1s - alpha2s
    denominator = 2 * I * sigma * np.sin(theta2)
    return numerator / denominator


def eq80(
    alpha1s: float, alpha2s: float, alpha3s: float, K: float, contrast: float
) -> Annotated[float, "2theta"]:
    de = 2 * np.sqrt(alpha1s * alpha2s)
    a = (alpha1s - alpha2s) / de
    b = (alpha1s + alpha2s) / de

    t = 2 / 5 * (contrast - 1)

    numerator = a * (1 + t / K * b**2)
    denominator = b * (1 + t / K * a**2)

    with warnings.catch_warnings(action="ignore"):
        return np.arccos(numerator / denominator)


class Roscoe:
    def __init__(self):
        pass

    def estimateMaximumDeformation(self, contrast: float) -> Tuple[float, float, float]:
        lowerAlpha1 = np.sqrt(2 / (contrast - 1.997) + 1)
        middleAlpha1 = np.sqrt(5.1 / (2 * contrast - 3.997) + 1)
        upperAlpha1 = np.sqrt(3.1 / (contrast - 2) + 1)

        return lowerAlpha1, middleAlpha1, upperAlpha1

    def prepare(self, contrast: Optional[float] = None, numberDatapoints: int = 10000):
        if contrast and contrast > 2:  # There is a maximum deformation
            l, m, u = self.estimateMaximumDeformation(contrast)
            l = 1  # We want our intervall to start at 1
            # Spend 90% of our resolution along the likely location of target alphas
            lowerIntervall = l - 1 + np.logspace(0, np.log10(m - l + 1), 9 * numberDatapoints // 10)
            upperIntervall = m - 1 + np.logspace(0, np.log10(u - m + 1), numberDatapoints // 10)
            self.alpha1s = (
                np.concatenate((lowerIntervall, upperIntervall[1:])) ** 2
            )  # Remove overlap
        else:
            # ALgorithm works until approx 1.7, but values will be smaller than 1
            # Otherwise change the value here
            self.alpha1s = np.logspace(0, 1, numberDatapoints) ** 2
        self.alpha2s = pooled(solve_eq78)(self.alpha1s)
        self.alpha3s = 1 / (self.alpha1s * self.alpha2s)

        self.I = eq39(self.alpha1s, self.alpha2s, self.alpha3s)
        self.K = eq43(self.alpha1s, self.alpha2s, self.alpha3s)
        self.prepared = True

    def calculate(self, contrast, sigma, shearRate, numberDatapoints: int = 10000):
        if not self.prepared:
            raise Exception("You need to prepare Roscoe before using it")
        theta2 = eq80(self.alpha1s, self.alpha2s, self.alpha3s, self.K, contrast)
        kappa = eq79(self.alpha1s, self.alpha2s, self.alpha3s, self.I, theta2, sigma)

        closest = np.nanargmin(np.abs(kappa - shearRate))

        kprox = kappa[closest]

        if closest == self.alpha1s.size - 1 and kprox < shearRate:
            raise Exception("Initial intervall too short")

        if kprox == shearRate:
            lower = closest
            upper = closest
        elif kprox > shearRate:
            lower = closest - 1
            upper = closest
        else:
            lower = closest
            upper = closest + 1

        err = (kappa[upper] - kappa[lower]) / shearRate

        a1s = self.alpha1s
        a2s = self.alpha2s
        a3s = self.alpha3s

        while err > 1e-4:  # Limit error
            l = np.sqrt(a1s[lower])
            u = np.sqrt(a1s[upper])
            if contrast > 2:
                a1s = (
                    np.logspace(
                        np.log10(l),
                        np.log10(u),
                        numberDatapoints,
                    )
                    ** 2
                )
            else:
                d = (u - l) / numberDatapoints
                a1s = (
                    np.arange(
                        l,
                        u * (1 + d),
                        d,
                    )
                    ** 2
                )
            a2s = pooled(solve_eq78)(a1s)
            a3s = 1 / (a1s * a2s)

            I = eq39(a1s, a2s, a3s)
            K = eq43(a1s, a2s, a3s)
            theta2 = eq80(a1s, a2s, a3s, K, contrast)
            kappa = eq79(a1s, a2s, a3s, I, theta2, sigma)
            closest = np.nanargmin(np.abs(kappa - shearRate))

            kprox = kappa[closest]

            if kprox == shearRate:
                lower = closest
                upper = closest
            elif kprox > shearRate:
                lower = closest - 1
                upper = closest
            else:
                lower = closest
                upper = closest + 1

            err = (kappa[upper] - kappa[lower]) / shearRate

        a1 = 0.5 * (np.sqrt(a1s[lower]) + np.sqrt(a1s[upper]))
        err1 = -(np.sqrt(a1s[lower]) - np.sqrt(a1s[upper]))
        a2 = 0.5 * (np.sqrt(a2s[lower]) + np.sqrt(a1s[upper]))
        err2 = -(np.sqrt(a2s[lower]) - np.sqrt(a1s[upper]))
        a3 = 0.5 * (np.sqrt(a3s[lower]) + np.sqrt(a1s[upper]))
        err3 = -(np.sqrt(a3s[lower]) - np.sqrt(a1s[upper]))
        t = 0.5 * (theta2[lower] + theta2[upper]) / 2
        errt = -(theta2[lower] - theta2[upper]) / 2
        k = 0.5 * (kappa[lower] + kappa[upper])
        errk = -(kappa[lower] - kappa[upper])

        return (a1, a2, a3, t, k), (err1, err2, err3, errt, errk)


__all__ = ["Roscoe"]
