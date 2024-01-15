# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Fri Jan 12 19:03:29 2024

@author: Richard Kellnberger
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import lambertw
from style import *
from style import cm

channel = 0
pipe = 1


class PTT:
    def __init__(self, eta_p, lambda_p, epsilon, eta_s, xi=0):
        self.eta_p = eta_p
        self.lambda_p = lambda_p
        self.epsilon = epsilon
        self.eta_s = eta_s
        self.xi = xi
        if xi != 0:
            print("xi != 0 is not implemented!")

    def prepareVelocityProfile(self, R, G, j):
        c = 2**j * self.eta_s / (-G)
        d = 2 * np.sqrt(self.epsilon) * self.lambda_p

        guesses = np.arange(0, R / c * (1 + 1e-6), R / c * 1e-6)

        def r(gd):
            return c * gd - self.eta_p / self.eta_s * c / d * np.sqrt(
                lambertw(d**2 * gd**2).real
            )

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
            self.eta_p / np.exp(0.5 * lambertw(4 * self.epsilon * (gd * self.lambda_p) ** 2).real)
            + self.eta_s
        )

    def N_1(self, gd):
        return (
            self.eta_p
            / (2 * self.epsilon * self.lambda_p)
            * lambertw(4 * self.epsilon * (self.lambda_p * gd) ** 2).real
        )

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


alginate = PTT(48.2, 0.343, 0.545, 1e-3)

R = 10e-6
G = 1e8  # Looks nice
G = 4e6  # Similar to steffen
G = 2e7  # Used in sim
alginate.prepareVelocityProfile(R, G, pipe)  # Q is per thickness for 2D case
print(alginate.Q)

µ = 1e6  # convert to µ
l = 1e1**3  # convert to liter
h = 1 / 3600  # convert to per hour
print(alginate.Q * µ * l / h)

r = np.arange(0, R * (1 + 1e-3), R * 1e-3)
print(r)
alginate.plot(r)

step = R / 20.5
points = np.arange(21)
print(points * step)

radii = []
for y in points:
    for z in points:
        if np.sqrt(y**2 + z**2) < 20.5:
            radii.append(np.sqrt(y**2 + z**2) * step)

gd = alginate.gd(np.asarray(radii))
N_1 = alginate.N_1(gd)
vs = alginate.eta(gd) * gd

plt.plot(radii, N_1)
plt.plot(radii, vs)
plt.show()

f = open("preset3D.csv", "w")
i = 0
for y in points:
    for z in points:
        if np.sqrt(y**2 + z**2) < 20.5:
            f.write(f"{y};{z};{N_1[i]:.20f};{vs[i]-1e-3*gd[i]:.20f}\n")
            i += 1
f.close()
