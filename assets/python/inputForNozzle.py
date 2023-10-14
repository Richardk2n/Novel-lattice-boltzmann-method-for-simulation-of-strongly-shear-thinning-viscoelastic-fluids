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


def process():
    baseDir = Path(f"../data/Nozzle/{0}/")
    cDir = baseDir / "vtkfiles/polymerConformationTensor/"
    vDir = baseDir / "vtkfiles/velocity/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    V0 = parameters["info"]["conversions"]["V0"]

    cFile = cDir / "C_5000000.vtk"
    vFile = vDir / "u_5000000.vtk"

    data = pv.read(cFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = data.dimensions
    cField = data.get_array("data")
    cField = np.reshape(cField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    out = open(f"../setups/Nozzle/2/input.tsv", "w")
    for y in range(Ly):
        for z in range(Lz):
            vx = vField[Lx // 2, y, z, 0]
            vy = vField[Lx // 2, y, z, 1]
            vz = vField[Lx // 2, y, z, 2]

            xx = cField[Lx // 2, y, z, 0]
            yy = cField[Lx // 2, y, z, 1]
            zz = cField[Lx // 2, y, z, 2]
            xy = cField[Lx // 2, y, z, 3]
            yz = cField[Lx // 2, y, z, 4]
            xz = cField[Lx // 2, y, z, 5]
            out.write(
                f"{1666}\t{y+(335-39)//2}\t{z+(335-39)//2}\t{vx}\t{vy}\t{vz}\t{xx}\t{yy}\t{zz}\t{xy}\t{yz}\t{xz}\n"
            )

    baseDir = Path(f"../data/Nozzle/{1}/")
    cDir = baseDir / "vtkfiles/polymerConformationTensor/"
    vDir = baseDir / "vtkfiles/velocity/"
    jsonFile = baseDir / "parameters.json"
    f = open(jsonFile)
    parameters = json.load(f)
    f.close()

    V0 = parameters["info"]["conversions"]["V0"]

    cFile = cDir / "C_5000000.vtk"
    vFile = vDir / "u_5000000.vtk"

    data = pv.read(cFile)
    dataV = pv.read(vFile)
    Lx, Ly, Lz = data.dimensions
    cField = data.get_array("data")
    cField = np.reshape(cField, (Lx, Ly, Lz, 6), "F")
    vField = dataV.get_array("data") * V0
    vField = np.reshape(vField, (Lx, Ly, Lz, 3), "F")

    for y in range(Ly):
        for z in range(Lz):
            vx = vField[Lx // 2, y, z, 0]
            vy = vField[Lx // 2, y, z, 1]
            vz = vField[Lx // 2, y, z, 2]

            xx = cField[Lx // 2, y, z, 0]
            yy = cField[Lx // 2, y, z, 1]
            zz = cField[Lx // 2, y, z, 2]
            xy = cField[Lx // 2, y, z, 3]
            yz = cField[Lx // 2, y, z, 4]
            xz = cField[Lx // 2, y, z, 5]
            out.write(f"{0}\t{y}\t{z}\t{vx}\t{vy}\t{vz}\t{xx}\t{yy}\t{zz}\t{xy}\t{yz}\t{xz}\n")
    out.close()


if __name__ == "__main__":
    t1 = time()
    process()
    print(time() - t1)
