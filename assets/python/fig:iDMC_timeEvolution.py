# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Sep  5 10:51:20 2023

@author: Richard Kellnberger
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from style import cm


def process(idx):
    baseDir = Path(f"../data/{idx}DParameterStudy/7/")
    vDir = baseDir / "vtkfiles/velocity/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    V0 = parameters["info"]["conversions"]["V0"]
    L0 = parameters["info"]["conversions"]["L0"]
    T0 = parameters["info"]["conversions"]["T0"]

    vs = []

    for i in range(101):
        vFile = vDir / f"u_{50000*i}.vtk"

        dataV = pv.read(vFile)
        Lx, Ly, Lz = dataV.dimensions
        vField = dataV.get_array("data") * V0
        vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

        vs.append(vField[Lx // 2, Ly // 2, Lz // 2, 0])

    plt.figure(figsize=(15.5 * cm, 15.5 / 2 * cm))
    plt.title("(b)", loc="left")
    plt.plot(np.arange(101) * 50000 * T0 * 1e3, np.asarray(vs))
    plt.xlabel(r"$t/\unit{\milli\second}$")
    plt.ylabel(r"$u_\text{x}/\unit{\meter\per\second}$")
    plt.savefig(f"../plots/{idx}DMC_timeEvolution.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    process(2)
    process(3)
