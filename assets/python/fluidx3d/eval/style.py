# -*- coding: utf-8 -*-
"""
File to house the style defenition.

Created on Fri Apr 14 14:49:37 2023

@author: Richard Kellnberger
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update(
    {
        "text.latex.preamble": "".join(
            [
                r"\usepackage[locale=US]{siunitx}"
                r"\sisetup{per-mode=fraction}"
                r"\sisetup{separate-uncertainty=true}"
                r"\newcommand{\ma}[1]{\underline{#1}}"
                r"\usepackage{amsmath}"
                r"\DeclareMathOperator{\Tr}{Tr}"
            ]
        )
    }
)
plt.rcParams["image.origin"] = "lower"

def rgb(r, g, b):
    return r / 255, g / 255, b / 255

tabColors = list(mcolors.TABLEAU_COLORS.keys())

def color(index):
    index %= len(tabColors)
    return tabColors[index]

backgroundColor = rgb(255, 255, 255)

def dark(r=51, g=51, b=51):
    global backgroundColor
    plt.style.use("dark_background")
    backgroundColor = rgb(r, g, b)
    plt.rcParams["figure.facecolor"] = rgb(r, g, b)
    plt.rcParams["axes.facecolor"] = rgb(r, g, b)

cm = 1 / 2.54
