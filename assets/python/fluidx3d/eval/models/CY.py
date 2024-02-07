# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Wed Feb  7 14:25:38 2024

@author: Richard Kellnberger
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import lambertw

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

    def prepareVelocityProfile(self, R, G, j):  # TODO
        c = 2**j * self.eta_s / (-G)
        d = 2 * np.sqrt(self.epsilon) * self.lambda_p

        guesses = np.arange(0, R / c * (1 + 1e-6), R / c * 1e-6)

        def r(gd):
            return c * gd - self.eta_p / self.eta_s * c / d * np.sqrt(lambertw(d**2 * gd**2).real)

        rguesses = r(guesses)
        idx = np.argmin(np.abs(rguesses - R))
        if rguesses[idx] < R:
            idx += 1

        guessesFine = np.arange(
            0, guesses[idx] * (1 + 3e-6), guesses[idx] * 1e-6
        )  # The three is due tue offsets needing it
        rguessesFine = r(guessesFine)

        rVelocity = 0.5 * (rguessesFine[:-1] + rguessesFine[1:])
        velocity = cumulative_trapezoid(guessesFine, rguessesFine)

        def getInterpolGD():
            def interpol(r):  # Can only handle scalar r
                idx = np.argmin(np.abs(rguessesFine - r))
                if rguessesFine[idx] > r:
                    idx -= 1
                elif rguessesFine[idx] == r:
                    return guessesFine[idx]
                m = (guessesFine[idx + 1] - guessesFine[idx]) / (
                    rguessesFine[idx + 1] - rguessesFine[idx]
                )
                return guessesFine[idx] + m * (r - rguessesFine[idx])

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

        def getInterpol(offset=0):
            def interpol(r):  # Can only handle scalar r
                idx = np.argmin(np.abs(rVelocity - r))
                if idx == 0:
                    m = (velocity[idx + 1] - velocity[idx]) / (rVelocity[idx + 1] - rVelocity[idx])
                    return velocity[idx] - m * (r - rVelocity[idx]) - offset
                if rVelocity[idx] > r:
                    idx -= 1
                elif rVelocity[idx] == r:
                    return velocity[idx] - offset
                m = (velocity[idx + 1] - velocity[idx]) / (rVelocity[idx + 1] - rVelocity[idx])
                return velocity[idx] + m * (r - rVelocity[idx]) - offset

            def interpolBroadcast(r):  # Nice, but needs ram
                idx = np.argmin(
                    np.abs(np.repeat([rVelocity], r.size, axis=0) - np.expand_dims(r, axis=1)),
                    axis=1,
                )
                idx -= rVelocity[idx] > r

                out = (
                    (idx == -1)
                    * (
                        velocity[idx]
                        - (
                            (velocity[idx + 2] - velocity[idx + 1])
                            / (rVelocity[idx + 2] - rVelocity[idx + 1])
                        )
                        * (r - rVelocity[idx])
                        - offset
                    )
                    + (rVelocity[idx] == r) * (velocity[idx] - offset)
                    + (1 - (idx == -1) * (rVelocity[idx] == r))
                    * (
                        velocity[idx]
                        + (
                            (velocity[idx + 1] - velocity[idx])
                            / (rVelocity[idx + 1] - rVelocity[idx])
                        )
                        * (r - rVelocity[idx])
                        - offset
                    )
                )
                return out

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
                np.concatenate(([0], rVelocity[:cutoff], [R])) ** j
                * np.concatenate(([self.u(0)], velocity[:cutoff] - offset, [0])),
                np.concatenate(([0], rVelocity[:cutoff], [R])),
            )
        )

    def eta(self, gd):
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

        plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
        plt.plot(r * 1e6, self.N_1(gd), label=r"$N_1$")
        plt.xlabel(r"$r/\unit{\micro\meter}$")
        plt.ylabel(r"$N_1/\unit{\pascal}$")
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


# alginate = PTT(48.2, 0.343, 0.545, 1e-3)
mc0_49 = CY(18.7e-3, 0.321e-3, 1e-3, 1.697, 0.773)
