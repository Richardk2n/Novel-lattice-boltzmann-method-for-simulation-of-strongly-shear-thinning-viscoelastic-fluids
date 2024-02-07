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
from fluidx3d.eval.models.Roscoe import Roscoe, modelRadiusSquared, modelX1, modelY1

R = 6
baseDir = Path("../data/cellInShear_2024-02-05/0/")
cDir = baseDir / "vtkfiles/cell/"
vtkIntervall = 4000


def process(i: int):
    cFile = cDir / f"cells_{vtkIntervall*i}.vtk"

    data = pv.read(cFile)
    points = data.points

    com = np.average(points, 0)

    p1 = (points[1] - com) / R
    p5 = (points[5] - com) / R

    return p1, p5


def evaluate():
    T0 = 4.34028e-10  # TODO read prom parameters.json
    # i = np.arange(13552)
    i = np.arange(27735)
    t = i * vtkIntervall * T0
    with Pool() as p:
        p1, p5 = np.swapaxes(p.map(process, i), 0, 1)

    extracted = extractRoscoe(t, p1, p5, 3000)

    (a1, a2, a3, th, ttf, phase), (err1, err2, err3, errth, errttf, errPhase) = extracted

    print(extracted)

    plt.plot(p1[..., 0] ** 2 + p1[..., 1] ** 2)
    plt.plot(modelRadiusSquared(t, a1**2, a2**2, ttf, phase))
    plt.show()

    plt.plot(p1[..., 0])
    plt.plot(modelX1(t, a1, a2, th, ttf, phase))
    plt.show()

    plt.plot(p1[..., 1])
    plt.plot(modelY1(t, a1, a2, th, ttf, phase))
    plt.show()

    theory = Roscoe()
    theory.prepare()
    shearRate = 2 * 0.2 / (100 * 1e-6)
    eta_0 = mc0_49.eta(shearRate)
    E = 1000
    poissonRatio = 0.48
    mu = E / (2 * (1 + poissonRatio))
    sigma = 2.5 * eta_0 / mu
    theo = theory.calculate(1, sigma, shearRate)
    print(theo)


if __name__ == "__main__":
    evaluate()
    """
    fit until 13552
    ((1.2480270945678713, 0.8261143904399001, 0.9749198, 0.5968765741610025, -1185.6738191957973, 5.542810806299429),
     (0.00014202758184686743, 0.00021278553442287667, 0.0010112725, 0.00017041897443233707, 0.06290683004371966, 0.0009521456788392505))
    ((1.2012968072932848, 0.829716787675813, 1.0032745597513146, 0.6044463844657088, 3999.99974499707, -1870.448988898043),
     (5.7571397781843916e-09, 5.964728205655835e-09, 2.075884197950652e-10, 1.0962578733675343e-08, 0.0002781878247333225, 0.00011455861420017754))
    """
