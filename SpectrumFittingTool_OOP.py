# -*- coding: utf-8 -*-
"""
Spectrum Fitting Tool
Author: ZHOU, Sai Kwan
Description: 
    An Python script to process optical spectroscopy data, 
    perform background subtraction, and execute automated multi-peak fitting.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op


class SpectrumAnalyzer:

    def __init__(self, data_path, bg_path):

        self.data_path = data_path
        self.bg_path = bg_path
        self.ept = None  # Number of data rows per spectrum spatial point

        # Instance variables to share data across different methods
        self.xx = None
        self.dt_data = None
        self.pk_results = None

    def gaussian2(self, x_array, amp1, cen1, sigma1, amp2, cen2, sigma2):

        g1 = amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (
            np.exp((-1.0 / 2.0) * (((x_array - cen1) / sigma1) ** 2))
        )
        g2 = amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (
            np.exp((-1.0 / 2.0) * (((x_array - cen2) / sigma2) ** 2))
        )
        return g1 + g2

    def load_and_preprocess(self):
        """
        Method 1: Load files, dynamically detect spectrum length (ept), and perform baseline subtraction.
        """
        print("[INFO] Loading data files...")
        bg_data = np.loadtxt(self.bg_path, skiprows=1)
        self.dt_data = np.loadtxt(self.data_path, skiprows=1)

        # Detect ept dynamically
        initial_x = self.dt_data[0, 0]
        initial_y = self.dt_data[0, 1]
        
        # Find all indices where coordinates differ from the first row
        coord_changes = np.where((self.dt_data[:, 0] != initial_x) | (self.dt_data[:, 1] != initial_y))[0]
        
        if len(coord_changes) > 0:
            self.ept = coord_changes[0]  
        else:
            self.ept = len(self.dt_data)  
            

        # Calculate the total number of background points and data points
        bgpt = int(len(bg_data) / self.ept)
        dtpt = int(len(self.dt_data) / self.ept)

        # Calculate the average background noise spectrum
        tbg_average = np.zeros([self.ept, 2])
        for j in range(bgpt):
            tbg_average += bg_data[self.ept * j : self.ept * (j + 1), 2:]
        tbg_average /= bgpt

        self.xx = tbg_average[:, 0]        # X-axis: Wavelength / Wavenumber
        bg_intensity = tbg_average[:, 1]   # Y-axis: Background baseline intensity

        # Subtract baseline noise from the core experimental data
        for x in range(dtpt):
            self.dt_data[self.ept * x : self.ept * (x + 1), 3] -= bg_intensity

        print(f"[INFO] Successfully preprocessed {dtpt} spatial map points.")

    def run_fitting(self, p0_init, fit_range, bounds):

        if self.ept is None:
            raise ValueError("[ERROR] Data not loaded yet. Call load_and_preprocess() first.")
        
        dtpt = int(len(self.dt_data) / self.ept)
        self.pk_results = np.zeros([dtpt, 6])

        print(
            f"[INFO] Starting fitting loop within range indices {fit_range}..."
        )
        for x in range(dtpt):
            yy = self.dt_data[self.ept * x : self.ept * (x + 1), 3]
            yy[yy < 0] = 0  # Physical constraint: Clip negative intensities to zero

            xx_fit = self.xx[fit_range[0] : fit_range[1]]
            yy_fit = yy[fit_range[0] : fit_range[1]]

            try:
                # Perform Levenberg-Marquardt least-squares curve fitting
                popt, _ = op.curve_fit(
                    self.gaussian2, xx_fit, yy_fit, p0=p0_init, bounds=bounds
                )
                self.pk_results[x] = popt

                # Periodically export plots to monitor convergence
                if x % 50 == 0:
                    plt.figure()
                    plt.plot(
                        self.xx,
                        yy,
                        "b.",
                        markersize=2,
                        label="Data (BG Subtracted)",
                    )
                    plt.plot(
                        self.xx, self.gaussian2(self.xx, *popt), "g-", label="Fit"
                    )
                    plt.title(f"Point {x} Fit Performance")
                    plt.legend()
                    plt.savefig(f"fit_result_point_{x}.png")
                    plt.close()

            except RuntimeError:
                print(f"[WARNING] Point {x} did not converge. Skipping...")
                self.pk_results[x] = np.nan

    def display_report(self):
        """
        Method 3: Conduct scientific statistics and print final analytics summary.
        """
        if self.pk_results is None:
            print("[ERROR] Execution halted. No fitting data available.")
            return

        ratio = self.pk_results[..., 0] / self.pk_results[..., 3]

        print("\n" + "=" * 50)
        print("          OPTICAL SPECTROSCOPY REPORT          ")
        print("=" * 50)
        print(
            f"Peak 1 Center Mean: {np.nanmean(self.pk_results[..., 1]):.2f} (SD: {np.nanstd(self.pk_results[..., 1]):.2f})"
        )
        print(
            f"Peak 2 Center Mean: {np.nanmean(self.pk_results[..., 4]):.2f} (SD: {np.nanstd(self.pk_results[..., 4]):.2f})"
        )
        print(
            f"Intensity Ratio Mean: {np.nanmean(ratio):.4f} (SD: {np.nanstd(ratio):.4f})"
        )
        print("=" * 50 + "\n")


if __name__ == "__main__":
    DATA_FILE = "DataFile.txt"
    BG_FILE = "BackgroundFile.txt"

    analyzer = SpectrumAnalyzer(data_path=DATA_FILE, bg_path=BG_FILE)
    analyzer.load_and_preprocess()

    # Define hyperparameters globally for external optimization and tuning
    p0_guesses = [5000, 1600, 10, 30000, 2700, 30]
    fit_window = [140, 900]
    param_bounds = [
        [100, 1550, 1, 5000, 2600, 1],  # Min constraints for [amp1, cen1, sig1, amp2, cen2, sig2]
        [30000, 1650, 30, 200000, 2750, 30],  # Max constraints
    ]

    # Run pipeline with custom tuning arguments
    analyzer.run_fitting(
        p0_init=p0_guesses, fit_range=fit_window, bounds=param_bounds
    )
    analyzer.display_report()