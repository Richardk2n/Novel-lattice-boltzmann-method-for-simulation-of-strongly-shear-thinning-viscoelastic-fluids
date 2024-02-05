# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Thu Jan 25 15:23:55 2024

@author: Richard Kellnberger
"""

import warnings
from functools import wraps
from multiprocessing import Pool
from typing import Annotated, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad  # type: ignore
from scipy.optimize import root_scalar  # type: ignore

floatN = Union[float, NDArray]


def pooled(fun):
    @wraps(fun)
    def decorated(*args: Tuple[NDArray]) -> NDArray:
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


def eq39(alpha1s: NDArray, alpha2s: NDArray, alpha3s: NDArray) -> NDArray:
    g1pp, g2pp, g3pp = np.transpose(pooled(eq18)(alpha1s, alpha2s, alpha3s))
    denominator: NDArray = g2pp * g3pp + g3pp * g1pp + g1pp * g2pp
    I = 0.4 * (g1pp + g2pp) / denominator  # noqa: E4731

    return I


def eq39_40(alpha1s: float, alpha2s: float, alpha3s: float) -> float:
    g1pp, g2pp = eq18r(alpha1s, alpha2s, alpha3s)

    return (g1pp - g2pp) / (g1pp + g2pp)


@overload
def eq41(
    alpha1s: float, alpha2s: float, theta2: float, kappa: float
) -> Annotated[float, "ttf"]: ...


@overload
def eq41(
    alpha1s: NDArray, alpha2s: NDArray, theta2: NDArray, kappa: NDArray
) -> Annotated[NDArray, "ttf"]: ...


def eq41(
    alpha1s: floatN, alpha2s: floatN, theta2: floatN, kappa: floatN
) -> Annotated[floatN, "ttf"]:
    return (
        -(alpha1s + alpha2s)
        / (2 * np.sqrt(alpha1s * alpha2s))
        * (1 - (alpha1s - alpha2s) / (alpha1s + alpha2s) * np.cos(theta2))
        * kappa
        / 2
    )


def eq43(alpha1s: NDArray, alpha2s: NDArray, alpha3s: NDArray) -> Annotated[NDArray, "K"]:
    return (alpha1s + alpha2s) / (5 * pooled(eq21)(alpha1s, alpha2s, alpha3s) * alpha1s * alpha2s)


def solve_eq78(alpha1s: float) -> Annotated[float, "alpha2s"]:
    def eq78(alpha2s: float) -> float:
        alpha3s: float = 1 / (alpha1s * alpha2s)
        JbyI = eq39_40(alpha1s, alpha2s, alpha3s)

        return JbyI * (alpha1s - alpha2s) - (alpha1s + alpha2s - 2 * alpha3s)

    sol = root_scalar(
        eq78,
        method="toms748",
        bracket=[1e-15, 1 / alpha1s],
    )

    if not sol.converged:
        print(f"{sol=}")
        raise Exception("Could not find alpha2")

    return sol.root


@overload
def eq79(
    alpha1s: float,
    alpha2s: float,
    alpha3s: float,
    I: float,  # noqa: E4731
    theta2: float,
    sigma: float,
) -> Annotated[float, "kappa"]: ...


@overload
def eq79(
    alpha1s: NDArray,
    alpha2s: NDArray,
    alpha3s: NDArray,
    I: NDArray,  # noqa: E4731
    theta2: NDArray,
    sigma: float,
) -> Annotated[NDArray, "kappa"]: ...


def eq79(
    alpha1s: floatN,
    alpha2s: floatN,
    alpha3s: floatN,
    I: floatN,  # noqa: E4731
    theta2: floatN,
    sigma: float,
) -> Annotated[floatN, "kappa"]:
    # typo in Roscoe!
    numerator = alpha1s - alpha2s
    denominator = 2 * I * sigma * np.sin(theta2)
    return numerator / denominator


@overload
def eq80(
    alpha1s: float, alpha2s: float, alpha3s: float, K: float, contrast: float
) -> Annotated[float, "2theta"]: ...


@overload
def eq80(
    alpha1s: NDArray, alpha2s: NDArray, alpha3s: NDArray, K: NDArray, contrast: float
) -> Annotated[NDArray, "2theta"]: ...


def eq80(
    alpha1s: floatN, alpha2s: floatN, alpha3s: floatN, K: floatN, contrast: float
) -> Annotated[floatN, "2theta"]:
    de = 2 * np.sqrt(alpha1s * alpha2s)
    a = (alpha1s - alpha2s) / de
    b = (alpha1s + alpha2s) / de

    t = 2 / 5 * (contrast - 1)

    numerator = a * (1 + t / K * b**2)
    denominator = b * (1 + t / K * a**2)

    with warnings.catch_warnings(action="ignore"):
        return np.arccos(numerator / denominator)


def modelRadiusSquared(
    t: NDArray, alpha1s: float, alpha2s: float, nu: float, phase: float
) -> NDArray:
    """
    Model used to evaluate roscoe simulations.

    This is equivalent to x**2 + y**2 for points at the surface at z = 0.
    Noteably does not contain theta and such is easiert to fit.

    Parameters
    ----------
    t : NDArray
        Time.
    alpha1s : float
        Squared alpha_1.
    alpha2s : float
        Squared alpha_2.
    nu : float
        Tank-threading frequency.
    phase : float
        Irrelevant.

    Returns
    -------
    NDArray
        x**2 + y**2.

    """
    return (alpha1s - alpha2s) * np.cos(nu * t + phase) ** 2 + alpha2s


def modelX1(
    t: NDArray, alpha1: float, alpha2: float, theta: float, nu: float, phase: float
) -> NDArray:
    """
    Model used to evaluete roscoe simulations.

    This models the x-coordinate for points at the surface at z = 0.

    Parameters
    ----------
    t : NDArray
        Time.
    alpha1 : float
        Squared alpha_1.
    alpha2 : float
        Squared alpha_2.
    theta : float
        Angle of inclation of the cell.
    nu : float
        Tank-threading frequency.
    phase : float
        Irrelevant.

    Returns
    -------
    NDArray
        x.

    """
    return alpha1 * np.cos(nu * t + phase) * np.cos(theta) - alpha2 * np.sin(
        nu * t + phase
    ) * np.sin(theta)


def modelY1(
    t: NDArray, alpha1: float, alpha2: float, theta: float, nu: float, phase: float
) -> NDArray:
    """
    Model used to evaluete roscoe simulations.

    This models the y-coordinate for points at the surface at z = 0.

    Parameters
    ----------
    t : NDArray
        Time.
    alpha1 : float
        Squared alpha_1.
    alpha2 : float
        Squared alpha_2.
    theta : float
        Angle of inclation of the cell.
    nu : float
        Tank-threading frequency.
    phase : float
        Irrelevant.

    Returns
    -------
    NDArray
        y.

    """
    return alpha1 * np.cos(nu * t + phase) * np.sin(theta) + alpha2 * np.sin(
        nu * t + phase
    ) * np.cos(theta)


class Roscoe:
    """
    Wrapper class to facilitate usage of the Roscoe theory without prior knowledge.

    Attributes
    ----------
    alpha1s : float
        Cached goemetry; square of alpha_1 guesses.
    alpha2s : float
        Cached goemetry; square of alpha_2 generated from alpha_1 guesses.
    alpha3s : float
        Cached goemetry; square of alpha_3 generated from alpha_1 guesses.
    I : float
        Cached goemetry; I (Roscoe eq. 39) calculated from alpha_1 guesses.
    K : float
        Cached goemetry; K (Roscoe eq. 43) calculated from alpha_1 guesses.
    prepared : bool
        Whether prepare was called.

    Methods
    -------
    estimateMaximumDeformation(contrast)
        Calculate maximum deformation for given contrast.
    prepare(self, contrast = None, numberDatapoints = 10000)
        Make initial guesses for alpha_1 and cache resulting geometry
    calculate(contrast, sigma, shearRate, numberDatapoints = 10000, maxRelativeError = 1e-4)
        Calculate the parameters returned by Roscoes theory.

    """

    def estimateMaximumDeformation(self, contrast: float) -> Tuple[float, float, float]:
        """
        Calculate maximum deformation for given contrast.

        For a given contrast > 2 there exists a maximum deformation (of alpha_1)
        This gives a lower, middle and upper estimate
        Lower and upper can be treated as bounds while middle is the best guess
        Valid for alpha1 < 10**1.7

        Parameters
        ----------
        contrast : float
            The viscosity contrast for which to calculate the estimates.

        Returns
        -------
        Tuple[float, float, float]
            lower middle and upper estimate.

        """
        lowerAlpha1 = np.sqrt(2 / (contrast - 1.997) + 1)
        middleAlpha1 = np.sqrt(5.1 / (2 * contrast - 3.997) + 1)
        upperAlpha1 = np.sqrt(3.1 / (contrast - 2) + 1)

        return lowerAlpha1, middleAlpha1, upperAlpha1

    def prepare(self, contrast: Optional[float] = None, numberDatapoints: int = 10000):
        """
        Prepare initial alpha_1 guesses and resulting geometry.

        As the geometry is not related to the parameters of the system, this can be calculated in
        advance. This can than be used for multiple lookups. One could consider turing this into a
        lookup table.

        The initial guesses are distributed linearly from alpha_1 = 1 to 10.
        If a contrast is specidied that would cause a maximum deformation to exist, the initial
        guesses are instead distributed up to the upper bound of the estimate for the maximum
        deformation. They are distributed along a logarithmic distributing centered in the middle
        estimate. This allows higher precission in this reagion. Do not set this if you plan to run
        the calculation with different contrasts.

        For the default number of datapoints this takes 2 s - 20 s depending on your computer.
        For the maximum performance this should be set at least at ten times your core count.

        Parameters
        ----------
        contrast : Optional[float], optional
            A contrast to determine sensible initial guesses. The default is None.
        numberDatapoints : int, optional
            The number of initial guesses. The default is 10000.

        Returns
        -------
        None.

        """
        if contrast and contrast > 2:  # There is a maximum deformation
            lo, m, u = self.estimateMaximumDeformation(contrast)
            lo = 1  # We want our intervall to start at 1
            # Spend 90% of our resolution along the likely location of target alphas
            lowerIntervall = (
                lo - 1 + np.logspace(0, np.log10(m - lo + 1), 9 * numberDatapoints // 10)
            )
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

        self.I = eq39(self.alpha1s, self.alpha2s, self.alpha3s)  # noqa: E4731
        self.K = eq43(self.alpha1s, self.alpha2s, self.alpha3s)
        self.prepared = True

    def calculate(
        self,
        contrast: float,
        sigma: float,
        shearRate: float,
        numberDatapoints: int = 10000,
        maxRelativeError: float = 1e-4,
    ) -> Tuple[
        Tuple[float, float, float, float, float, float],
        Tuple[float, float, float, float, float, float],
    ]:
        """
        Calculate the parameters returned by Roscoes theory.

        This is run iteratively until the given error is reached. The resulting error might be
        significantly better depending on the number of datapoints in each step. For the default
        value a step takes 2 s - 20 s depending on your computer. For the maximum performance this
        should be set at least at ten times your core count.

        The mu in sigma is the shear modulus.

        Parameters
        ----------
        contrast : float
            The viscosity contrast.
        sigma : float
            2.5*eta_0/mu as specified by roscoe eq. 62.
        shearRate : float
            The shear rate applied to the cell.
        numberDatapoints : int, optional
            The number of new points if a more refined guess is required. The default is 10000.
        maxRelativeError : float, optional
            How far the solution may be from the exact given shear rate. The default is 1e-4.

        Raises
        ------
        Exception
            If you forgot to prepare or if the initial intervall is too short.

        Returns
        -------
        Tuple[Tuple[float, float, float, float, float, float], Tuple[float, float, float, float, float, float]] # noqa: E501
            alpha_1, alpha_2, alpha_3, theta, kappa, nu and their absolute errors.

        """
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

        # This error definition carries the assumption,
        # that kappa is essentially linear between upper and lower
        # This is wrong given large enough steps
        err = np.abs((kappa[upper] + kappa[lower] - 2 * shearRate) / 2 / shearRate)

        a1s = self.alpha1s
        a2s = self.alpha2s
        a3s = self.alpha3s

        while err > maxRelativeError:  # Limit error
            lo = np.sqrt(a1s[lower])
            u = np.sqrt(a1s[upper])
            if contrast > 2:
                a1s = (lo - 1 + np.logspace(0, np.log10(u - lo + 1), numberDatapoints)) ** 2
            else:
                d = (u - lo) / numberDatapoints
                a1s = np.arange(lo, u * (1 + d), d) ** 2
            a2s = pooled(solve_eq78)(a1s)
            a3s = 1 / (a1s * a2s)

            I = eq39(a1s, a2s, a3s)  # noqa: E4731
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

            # This error definition carries the assumption,
            # that kappa is essentially linear between upper and lower
            # This is wrong given large enough steps
            err = np.abs((kappa[upper] + kappa[lower] - 2 * shearRate) / 2 / shearRate)

        nu = eq41(a1s, a2s, theta2, kappa)  # Calculating this for each point is unnecessary

        # Error propagation for the sequared alphas
        a1 = 0.5 * (np.sqrt(a1s[lower]) + np.sqrt(a1s[upper]))
        err1 = np.abs(np.sqrt(a1s[lower]) - np.sqrt(a1s[upper])) / 2 / (2 * a1)
        a2 = 0.5 * (np.sqrt(a2s[lower]) + np.sqrt(a2s[upper]))
        err2 = np.abs(np.sqrt(a2s[lower]) - np.sqrt(a2s[upper])) / 2 / (2 * a2)
        a3 = 0.5 * (np.sqrt(a3s[lower]) + np.sqrt(a3s[upper]))
        err3 = np.abs(np.sqrt(a3s[lower]) - np.sqrt(a3s[upper])) / 2 / (2 * a3)
        t = 0.5 * (theta2[lower] + theta2[upper]) / 2
        errt = np.abs(theta2[lower] - theta2[upper]) / 2 / 2
        k = 0.5 * (kappa[lower] + kappa[upper])
        errk = np.abs(kappa[lower] - kappa[upper]) / 2
        ttf = 0.5 * (nu[lower] + nu[upper])
        errttf = np.abs(nu[lower] - nu[upper]) / 2

        return (a1, a2, a3, t, k, ttf), (err1, err2, err3, errt, errk, errttf)

    def getPreparedInternals(
        self, contrast: float, sigma: float
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Return all values for the prepared points.

        Usefull to evaluate the behavior of the underlying equations.
        Can replace calculate in some usecases if high accuray is optional.

        The mu in sigma is the shear modulus.

        Parameters
        ----------
        contrast : float
            The viscosity contrast..
        sigma : float
            2.5*eta_0/mu as specified by roscoe eq. 62.

        Raises
        ------
        Exception
            If you forgot to prepare or if the initial intervall is too short.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
            alpha_1, alpha_2, alpha_3, theta, kappa, nu.

        """
        if not self.prepared:
            raise Exception("You need to prepare Roscoe before using it")
        theta2 = eq80(self.alpha1s, self.alpha2s, self.alpha3s, self.K, contrast)
        kappa = eq79(self.alpha1s, self.alpha2s, self.alpha3s, self.I, theta2, sigma)
        nu = eq41(self.alpha1s, self.alpha2s, theta2, kappa)
        return (
            np.sqrt(self.alpha1s),
            np.sqrt(self.alpha2s),
            np.sqrt(self.alpha3s),
            theta2 / 2,
            kappa,  # Shear rate
            nu,  # Tank-treading frequency
        )


# Only Roscoe class and models are for public consumption
__all__ = ["Roscoe", "modelRadiusSquared", "modelX1", "modelY1"]
