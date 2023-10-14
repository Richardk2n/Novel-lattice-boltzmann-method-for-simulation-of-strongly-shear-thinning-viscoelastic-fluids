# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy
from scipy.special import lambertw
from style import *

eta_s = 1e-3
j = 1
# R = 200e-6  # G = 77891.69129767214
R = 1.8e-3  # G = 11.947169068625888


def ptt_eta(gd, eta_p, lam, epsilon):
    return eta_p / np.exp(0.5 * lambertw(4 * epsilon * (gd * lam) ** 2)) + eta_s


def getl2(G, eta_p, lam, epsilon):
    return np.float128(2 ** (2 * j - 1) * eta_p**2 / (epsilon * lam**2 * G**2))


# According to Oliveira 1999
def uS(r, G, eta_p, lam, epsilon):  # eta_s = 0
    l2 = getl2(G, eta_p, lam, epsilon)
    return G * l2 / (2 ** (j + 1) * (eta_p + eta_s)) * (np.exp(R**2 / l2) - np.exp(r**2 / l2))


def uHP(r, G, eta_p):
    return G / (4 * (eta_p + eta_s)) * (R**2 - r**2)


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


def calcQ(G):
    eta_p_SI = 18.7e-3
    lambda_SI = 0.344e-3
    epsilon = 0.27

    rsa, usa = findRoot(G, lambda_SI, eta_p_SI, epsilon)

    rsa = np.concatenate((-np.flip(rsa[1:]), rsa))
    usa = np.concatenate((np.flip(usa[1:]), usa))

    dr = rsa[1] - rsa[0]
    q = 2 * np.pi * np.sum(usa * np.abs(rsa)) * dr
    return q * 1e3 * 1e6  # µl/s


def find(G):
    return calcQ(G) - 5  # Search for 5µl/s


def process():
    answ = scipy.optimize.root(find, 1e2)
    print(answ)
    print(f"needed G: {answ.x[0]}")


if __name__ == "__main__":
    t1 = time()
    process()
    print("time:")
    print(time() - t1)
