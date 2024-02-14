# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Wed Feb  7 14:25:38 2024

@author: Richard Kellnberger
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import root_scalar  # type: ignore

from fluidx3d.eval.style import cm

channel = 0
pipe = 1


class CY:
    def __init__(self, eta_p, lambda_p, eta_s, a1, a2):
        self.eta_p = eta_p
        self.lambda_p = lambda_p
        self.eta_s = eta_s
        self.a1 = a1
        self.a2 = a2

    def prepareVelocityProfile(self, R, G, j):  # TODO check j
        stepSize = 1e-6

        def stressBalance(gd):
            return self.eta(gd) * gd + G / 2**j * R

        sol = root_scalar(
            stressBalance,
            method="toms748",
            bracket=[-G / 2**j * R / self.eta_s, -G / 2**j * R / (self.eta_p + self.eta_s)],
        )

        if not sol.converged:
            print(f"{sol=}")
            raise Exception("Could not find alpha2")

        edgeGD = sol.root

        gds = np.arange(0, 1 + stepSize, stepSize) * edgeGD
        rs = -2 / G * self.eta(gds) * gds

        def getInterpolGD():
            def interpol(r):  # Can only handle scalar r
                idx = np.argmin(np.abs(rs - r))
                if (rs[idx] > r and idx != 0) or idx == rs.shape[0] - 1:  # Make one sided instead
                    idx -= 1
                elif rs[idx] == r:
                    return gds[idx]
                m = (gds[idx + 1] - gds[idx]) / (rs[idx + 1] - rs[idx])
                return gds[idx] + m * (r - rs[idx])

            def interpolLoop(r):  # Slow Python loop
                out = []
                for i in range(r.size):
                    out.append(interpol(r[i]))
                return np.asarray(out)

            def selector(r):
                if np.shape(r) == ():
                    return interpol(r)
                else:
                    return interpolLoop(r)

            return selector

        self.gd = getInterpolGD()

        rv = 0.5 * (rs[:-1] + rs[1:])
        velocity = cumulative_trapezoid(gds, rs)

        def getInterpol(offset=0):
            def interpol(r):  # Can only handle scalar r
                idx = np.argmin(np.abs(rv - r))
                if (rv[idx] > r and idx != 0) or idx == rv.shape[0] - 1:  # Make one sided instead
                    idx -= 1
                elif rv[idx] == r:
                    return velocity[idx] - offset
                m = (velocity[idx + 1] - velocity[idx]) / (rv[idx + 1] - rv[idx])
                return velocity[idx] + m * (r - rv[idx]) - offset

            def interpolLoop(r):  # Slow Python loop
                out = []
                for i in range(r.size):
                    out.append(interpol(r[i]))
                return np.asarray(out)

            def selector(r):
                if np.shape(r) == ():
                    return interpol(r)
                else:
                    return interpolLoop(r)

            return selector

        offset = getInterpol()(R)
        self.u = getInterpol(offset)

        cutoff = np.argmax((velocity - offset) < 0)

        self.Q = (
            2
            * np.pi**j
            * trapezoid(
                np.concatenate(([0], rv[:cutoff], [R])) ** j
                * np.concatenate(([self.u(0)], velocity[:cutoff] - offset, [0])),
                np.concatenate(([0], rv[:cutoff], [R])),
            )
        )

    def eta(self, gd):
        gd = np.abs(gd)
        return (
            self.eta_p / (1 + (self.lambda_p * gd) ** self.a1) ** (self.a2 / self.a1) + self.eta_s
        )

    def u(self, _):
        raise Exception("Forgot to call prepareVelocityProfile before u")

    def gd(self, _):
        raise Exception("Forgot to call prepareVelocityProfile before gd")

    def plot(self, r):
        u = self.u(r)

        plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
        plt.plot(r * 1e6, u, label=r"$v_x$")
        plt.xlabel(r"$r/\unit{\micro\meter}$")
        plt.ylabel(r"$v_x/\unit{\meter\per\second}$")
        plt.legend()
        plt.show()

        gd = self.gd(r)

        plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
        plt.plot(r * 1e6, gd, label=r"$\dot{\gamma}$")
        plt.xlabel(r"$r/\unit{\micro\meter}$")
        plt.ylabel(r"$\dot{\gamma}/\unit{\per\second}$")
        plt.legend()
        plt.show()

        eta = self.eta(gd)
        plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
        plt.plot(r * 1e6, eta, label=r"$\eta$")
        plt.xlabel(r"$r/\unit{\micro\meter}$")
        plt.ylabel(r"$\eta/\unit{\pascal\second}$")
        plt.legend()
        plt.show()

        rho = 1e3
        Re = rho * u * np.max(r) / eta

        plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
        plt.plot(r * 1e6, Re, label=r"$Re$")
        plt.xlabel(r"$r/\unit{\micro\meter}$")
        plt.ylabel(r"$Re$")
        plt.legend()
        plt.show()

        plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
        plt.plot(r * 1e6, eta * gd, label=r"$\sigma$")
        plt.xlabel(r"$r/\unit{\micro\meter}$")
        plt.ylabel(r"$\sigma/\unit{\pascal}$")
        plt.legend()
        plt.show()


# alginate = PTT(48.2, 0.343, 0.545, 1e-3)
mc0_49 = CY(18.7e-3, 0.321e-3, 1e-3, 1.697, 0.773)
