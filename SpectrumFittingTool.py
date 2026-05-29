# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:42:24 2023

@author: zhous
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op


# 雙高斯擬合函數
def gaussian2(x_array, amp1, cen1, sigma1, amp2, cen2, sigma2):
    return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (
        np.exp((-1.0 / 2.0) * (((x_array - cen1) / sigma1) ** 2))
    ) + amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (
        np.exp((-1.0 / 2.0) * (((x_array - cen2) / sigma2) ** 2))
    )



print("Loading data files...")
BG_data = np.loadtxt(
    "BackgroundFile.txt",
    skiprows=1,
)
DT_data = np.loadtxt(
    "DataFile.txt",
    skiprows=1,
)

ept = 1015  #eachpointsdata
bgpt = int(len(BG_data) / ept)  #bgpoints
dtpt = int(len(DT_data) / ept)  #dtpoints

#averagebg
TBGA = np.zeros([ept, 2])
for j in range(bgpt):
    TBGA += BG_data[ept * j : ept * (j + 1), 2:]
TBGA /= bgpt

#reducebg
bg_intensity = TBGA[:, 1]
for x in range(dtpt):
    DT_data[ept * x : ept * (x + 1), 3] -= bg_intensity

#parameters
p0A = [5000, 1600, 10, 30000, 2700, 30]
p1 = [140, 900]
bound = [[100, 1550, 1, 5000, 2600, 1], [30000, 1650, 30, 200000, 2750, 30]]

xx = TBGA[:, 0]
PKDT = np.zeros([dtpt, 6])

print("Starting curve fitting for all points...")

for x in range(dtpt):
    yy = DT_data[ept * x : ept * (x + 1), 3]
    yy[yy < 0] = 0  

    xx_fit = xx[p1[0] : p1[1]]
    yy_fit = yy[p1[0] : p1[1]]

    try:
        popt1, pcov1 = op.curve_fit(
            gaussian2, xx_fit, yy_fit, p0=p0A, bounds=bound
        )
        PKDT[x] = popt1

        if x % 50 == 0:
            plt.figure()
            plt.plot(xx, yy, "b.", label="Data")
            plt.plot(xx, gaussian2(xx, *popt1), "g-", label="Fit")
            plt.title(f"Coord: {DT_data[ept*x, :2]}")
            plt.legend()
            plt.savefig(f"fit_result_point_{x}.png")
            plt.close()

    except RuntimeError:
        print(f"Warning: Point {x} fit did not converge.")
        PKDT[x] = np.nan

#Outputs
IR = PKDT[..., 0] / PKDT[..., 3]
print("\n=== FIT SUMMARY ===")
print("Peak 1 mean:", np.nanmean(PKDT[..., 1]), ", SD:", np.nanstd(PKDT[..., 1]))
print("Peak 2 mean:", np.nanmean(PKDT[..., 4]), ", SD:", np.nanstd(PKDT[..., 4]))
print("P1:P2 Mean intensity ratio:", np.nanmean(IR), ", SD:", np.nanstd(IR))