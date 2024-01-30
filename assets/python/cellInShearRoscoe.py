# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Jan 30 12:59:02 2024

@author: Richard Kellnberger
"""

from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from fluidx3d.eval.extract.RoscoeFit import extractRoscoe
from fluidx3d.eval.models.PTT import mc0_49
from fluidx3d.eval.models.Roscoe import Roscoe, modelRadiusSquared, modelX1

R = 6
baseDir = Path("../data/cellInShear_2024-01-23/0/")
cDir = baseDir / "vtkfiles/cell/"
vtkIntervall = 4000


def process(i: int):
    cFile = cDir / f"cells_{vtkIntervall*i}.vtk"

    data = pv.read(cFile)
    points = data.points

    p1 = points[1] / R
    p5 = points[5] / R

    return p1, p5


def evaluate():
    T0 = 4.34028e-09
    i = np.arange(10001)
    t = i * vtkIntervall * T0
    with Pool() as p:
        p1, p5 = np.swapaxes(p.map(process, i), 0, 1)

    extracted = extractRoscoe(t, p1, p5, 2000)

    (a1, a2, a3, th, ttf, phase), (err1, err2, err3, errth, errttf, errPhase) = extracted

    print(extracted)

    plt.plot(p1[..., 0] ** 2 + p1[..., 1] ** 2)
    plt.plot(modelRadiusSquared(t, a1**2, a2**2, ttf, phase))
    plt.show()

    plt.plot(p1[..., 0])
    plt.plot(modelX1(t, a1, a2, th, ttf, phase))
    plt.show()

    theory = Roscoe()
    theory.prepare()
    shearRate = 2 * 0.02 / (100 * 1e-6)
    eta_0 = mc0_49.eta(shearRate)
    E = 100
    poissonRatio = 0.48
    mu = E / (2 * (1 + poissonRatio))
    sigma = 2.5 * eta_0 / mu
    theo = theory.calculate(1, sigma, shearRate)
    print(theo)


if __name__ == "__main__":
    evaluate()
    """
    ((1.3248546911706085, 0.7672914055519995, 1.0040803, 0.5079350485998886, -173.79061966102753, 5.6818158441716395),
     (0.00012980915553869758, 0.0002144417488200787, 0.0006408331, 0.00020270748456399561, 0.00570695540593391, 0.000636761329030773))
    ((1.2950496110600689, 0.7671202453877896, 1.0065842939194285, 0.5347741444098003, 400.00000708330896, -175.39671056727622),
     (1.4910436330062282e-08, 9.288833036080746e-09, 5.992155660550225e-10, 1.0358102930929647e-08, 2.139741894779945e-05, 7.391680057367012e-06))
    """
