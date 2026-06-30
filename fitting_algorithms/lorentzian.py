import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import lorentzian_config as lc


class Lorentzian:
    def __init__(self, naive, adaptive_sweep_points, adaptive_iterations, fetch_data):
        self.naive = naive
        self.adaptive_sweep_points = adaptive_sweep_points
        self.adaptive_iterations = adaptive_iterations
        self.fetch_data = fetch_data

        self.transmission_coupled = lc.TRANSMISSION_COUPLED
        self.f0_approx = None
        self.kappa_approx = None

        self.initial_freqs = None
        self.background_est = None

    @staticmethod
    def transmission_lorentzian(f, f_r, kappa, baseline, separation):
        complex_s21 = baseline + separation / (1 + 1j * 2 * (f - f_r) / kappa)
        return np.abs(complex_s21)
    
    @staticmethod
    def side_coupled_lorentzian(f, kappa, separation):
        abs_s21 = -1 * separation * (kappa ** 2 /((f - lc.TARGET_F0)**2 + kappa**2))
        return abs_s21

    def generate_new_freqs(self):
        epsilon = lc.GENERATE_NEW_FREQS_EPSILON
        theta = np.linspace(np.pi/2 - epsilon, -np.pi/2 + epsilon, self.adaptive_sweep_points)
        new_freqs = self.f0_approx + self.kappa_approx * np.tan(theta)
        new_freqs = new_freqs[(new_freqs <= lc.GENERATE_NEW_FREQS_UPPER_LIMIT) & (new_freqs >= lc.GENERATE_NEW_FREQS_LOWER_LIMIT)]
        return np.sort(new_freqs)

    def clean_data(self, freq, s21, k_est=lc.PRIOR_KAPPA):
        span = np.abs(freq[-1] - freq[0])
        s21_mag = s21
        n = len(s21_mag)

        course = int(1*np.ceil(n*100*k_est/(span)))
        if course > np.ceil(2*n/3):
            course = n
        if course % 2 == 0: course += 1

        fine = int(np.ceil(n*20*k_est/(span)))
        if course == n: 
            fine = int(np.ceil(n*5*k_est/(span)))
        if fine > n/10: fine = int(n/10)
        if fine < 6: fine = 6
        if fine % 2 == 0: fine += 1

        background_est = savgol_filter(s21_mag, window_length= course, polyorder=2)
        s21_mag_corrected = s21_mag - background_est
        s21_mag_corrected_2 =  savgol_filter(s21_mag_corrected, window_length=fine, polyorder=2)
        return freq, s21_mag_corrected_2,background_est

    def subtract_local_linear_batch(self, freq_bg, bg_est, freq_new, s21_new):
        freq_bg = np.asarray(freq_bg)
        bg_est = np.asarray(bg_est)

        freq_new = np.asarray(freq_new)
        s21_new = np.asarray(s21_new)

        corrected_vals = []
        bg_interp_vals = []

        for f_new, s_new in zip(freq_new, s21_new):
            idx = np.searchsorted(freq_bg, f_new)

            if idx == 0:
                i0, i1 = 0, 1
            elif idx >= len(freq_bg):
                i0, i1 = -2, -1
            else:
                i0, i1 = idx - 1, idx

            x0, x1 = freq_bg[i0], freq_bg[i1]
            y0, y1 = bg_est[i0], bg_est[i1]
            
            t = (f_new - x0) / (x1 - x0)
            bg_interp = y0 + t * (y1 - y0)

            corrected = s_new - bg_interp

            corrected_vals.append(corrected)
            bg_interp_vals.append(bg_interp)

        return np.array(corrected_vals)

    def fit(self, freqs, s21_mag):
        if self.transmission_coupled:
            separation_guess = np.abs(np.max(s21_mag))
            kappa_guess = np.abs((freqs[-1] - freqs[0])) / 30.0

            p0 = [kappa_guess, separation_guess]
            bounds = ([0., 0], [1.0e6, np.inf])

            popt, _ = curve_fit(self.transmission_lorentzian, freqs, s21_mag, p0=p0, bounds=bounds, maxfev=lc.SCIPY_MAXFEV)
            return popt
        else:
            separation_guess = np.abs(np.min(s21_mag))
            kappa_guess = np.abs((freqs[-1] - freqs[0])) / 30.0

            p0 = [kappa_guess, separation_guess]
            bounds = ([0., 0], [100.0e6, np.inf])

            popt, _ = curve_fit(self.side_coupled_lorentzian, freqs, s21_mag, p0=p0, bounds=bounds, maxfev=lc.SCIPY_MAXFEV)
            return popt
        
    def raw_vs_clean_plot(self, freqs, s21_raw_mag, s21_clean_mag, initial_freqs, bg_est, title_suffix="Dense"):
        plt.figure(figsize=(10, 6))
        plt.title(f"Raw vs Cleaned Data Plot: {title_suffix}")
        plt.plot(freqs, s21_raw_mag, 'b.-', label='Raw S21')
        plt.plot(freqs, s21_clean_mag, 'g.-', label='Cleaned S21')
        plt.plot(initial_freqs, bg_est, 'r.-', label='Background Estimate')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('|S21|')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        return plt.gcf()

    def data_vs_fit_plot(self, freqs, s21_mag, popt, title_suffix="Dense"):    
        freqs_fit = np.linspace(freqs.min(), freqs.max(), int(lc.PLOT_POINTS))
        s21_fit = self.transmission_lorentzian(freqs_fit, *popt) if self.transmission_coupled else self.side_coupled_lorentzian(freqs_fit, *popt)
        plt.figure(figsize=(10, 6))
        plt.title(f"Data vs Fit Plot: {title_suffix}")
        plt.plot(freqs, s21_mag, "o-", markersize=4, label="Data")
        plt.plot(freqs_fit, s21_fit, "r-", label="Fit")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("|S21|")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        return plt.gcf()
    
    def fitting_routine(self):
        freqs, s21_complex = self.fetch_data()
        s21_mag = np.abs(s21_complex)
        freqs, s21_mag_clean, background_est = self.clean_data(freqs, s21_mag)
        
        self.initial_freqs = freqs.copy()
        self.background_est = background_est.copy()

        self.f0_approx = freqs[np.argmin(s21_mag_clean)]
        lc.TARGET_F0 = self.f0_approx
        popt = self.fit(freqs, s21_mag_clean)
        self.kappa_approx = popt[0]
        print(f"f0 = {self.f0_approx/1e9:.4f} GHz, kappa = {np.abs(popt[0])/1e6:.4f} MHz\n")

        plots_dict = {}
        cleaned_data_dict = {}
        
        base_name = "dense" if self.naive else "coarse"

        plots_dict[f"0_{base_name}"] = self.data_vs_fit_plot(freqs, s21_mag_clean, popt, title_suffix=base_name.capitalize())
        plots_dict[f"0_{base_name}_raw_vs_clean"] = self.raw_vs_clean_plot(freqs, s21_mag, s21_mag_clean, self.initial_freqs, self.background_est, title_suffix=base_name.capitalize())
        
        cleaned_data_dict[f"0_{base_name}"] = {"freqs": freqs.copy(), "s21_clean": s21_mag_clean.copy()}
        cleaned_data_dict["bg_est"] = {"freqs": freqs.copy(), "bg_est_data": background_est.copy()}

        if not self.naive:
            for i in range(self.adaptive_iterations):
                new_sweep_freqs = self.generate_new_freqs()

                freqs, s21_complex = self.fetch_data(new_sweep_freqs)
                s21_mag = np.abs(s21_complex)
                s21_mag_clean = self.subtract_local_linear_batch(freq_bg=self.initial_freqs, bg_est=self.background_est, freq_new=freqs, s21_new=s21_mag)

                popt = self.fit(freqs, s21_mag_clean)
                self.f0_approx = freqs[np.argmin(s21_mag_clean)]
                lc.TARGET_F0 = self.f0_approx
                self.kappa_approx = popt[0]
                print(f"f0 = {self.f0_approx/1e9:.4f} GHz, kappa = {np.abs(popt[0])/1e6:.4f} MHz\n")

                plots_dict[f"{i+1}_adaptive"] = self.data_vs_fit_plot(freqs, s21_mag_clean, popt, title_suffix=f"Adaptive {i+1}")
                plots_dict[f"{i+1}_adaptive_raw_vs_clean"] = self.raw_vs_clean_plot(freqs, s21_mag, s21_mag_clean, self.initial_freqs, self.background_est, title_suffix=f"Adaptive {i+1}")
                
                cleaned_data_dict[f"{i+1}_adaptive"] = {"freqs": freqs.copy(), "s21_clean": s21_mag_clean.copy()}

        return self.f0_approx, self.kappa_approx, plots_dict, cleaned_data_dict