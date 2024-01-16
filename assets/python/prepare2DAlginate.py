# -*- coding: utf-8 -*-
"""
File to house the   class.

Created on Tue Jan 16 20:35:58 2024

@author: Richard Kellnberger
"""

# 2DAlginate is 2024-01-15/0

import numpy as np
import matplotlib.pyplot as plt
from fluidx3d.eval.models import PTT

R = 10e-6
# G = 1e8  # Looks nice
# G = 4e6  # Similar to steffen
G = 1e7  # Used in sim
PTT.alginate.prepareVelocityProfile(R, G, PTT.channel)  # Q is per thickness for 2D case
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

gd = PTT.alginate.gd(points * step)
N_1 = PTT.alginate.N_1(gd)
vs = PTT.alginate.eta(gd) * gd

plt.plot(points * step, N_1)
plt.plot(points * step, vs)
plt.show()

f = open("preset.csv", "w")
for i in points:
    f.write(f"{i};{N_1[i]:.20f};{vs[i]-1e-3*gd[i]:.20f}\n")
f.close()
