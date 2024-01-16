# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Tue Jan 16 20:35:58 2024

@author: Richard Kellnberger
"""

# 3DAlginate is 2024-01-15/1

import numpy as np
import matplotlib.pyplot as plt
from fluidx3d.eval.models import PTT

R = 10e-6
# G = 1e8  # Looks nice
# G = 4e6  # Similar to steffen
G = 2e7  # Used in sim
PTT.alginate.prepareVelocityProfile(R, G, PTT.pipe)  # Q is per thickness for 2D case
print(PTT.alginate.Q)

µ = 1e6  # convert to µ
l = 1e1**3  # convert to liter
h = 1 / 3600  # convert to per hour
print(PTT.alginate.Q * µ * l / h)

r = np.arange(0, R * (1 + 1e-3), R * 1e-3)
print(r)
PTT.alginate.plot(r)

step = R / 20.5
points = np.arange(21)
print(points * step)

radii = []
for y in points:
    for z in points:
        if np.sqrt(y**2 + z**2) < 20.5:
            radii.append(np.sqrt(y**2 + z**2) * step)

gd = PTT.alginate.gd(np.asarray(radii))
N_1 = PTT.alginate.N_1(gd)
vs = PTT.alginate.eta(gd) * gd

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
