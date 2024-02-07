# -*- coding: utf-8 -*-
"""
File to house the  class.

Created on Tue Feb  6 15:58:51 2024

@author: Richard Kellnberger
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from fluidx3d.eval.models.PTT import mc0_49

gd = np.logspace(0, 10, 1000)
eta = mc0_49.eta(gd)


def CY(gd, lambda_p, a1, a2):  # Fix viscosities
    return mc0_49.eta_p / (1 + (lambda_p * gd) ** a1) ** (a2 / a1) + mc0_49.eta_s


p, cov = curve_fit(CY, gd, eta, [0.5 * mc0_49.lambda_p, 1, 1])
err = np.sqrt(np.diag(cov))
print(p)
print(err)

"""
[3.21435421e-04 1.69720860e+00 7.73490872e-01]
[1.19610108e-06 4.60219462e-03 1.49667878e-03]
Use as:
(0.321 +- 0.001)1e-3
1.697 +- 0.005
0.773 +- 0.001
"""

# One could argue that fitting the lof curve would be better. In fact the discrapency for high gd
# reduces while the one at the first bend increases. As this es our primary area of interest we do
# not perform a log fit

plt.plot(gd, eta)
plt.plot(gd, CY(gd, *p))
plt.xscale("log")
plt.yscale("log")
plt.show()
