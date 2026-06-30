import os
import time
import numpy as np
import matplotlib.pyplot as plt
import config
from instruments.vna import VNA
from instruments.hdawg import HDAWG as SG
from instruments.attenuator import Attenuator # Added
from fitting_algorithms.lorentzian import Lorentzian


class QubitScan:
    def __init__(self, naive, readout_frequency, readout_power, vna_ip, vna_timeout_ms, vna_power, vna_ifbw, vna_averages, start_freq, stop_freq, coarse_sweep_points, dense_sweep_points, adaptive_sweep_points, adaptive_iterations, sg_powers, labber_server, labber_server_timeout, attenuator_name, attenuator_interface, attenuator_address, attenuator_wait_s, raw_data_dir, plots_dir):
        self.naive = naive
        self.readout_frequency = readout_frequency
        self.readout_power = readout_power
        self.vna_power = vna_power
        self.vna_ifbw = vna_ifbw
        self.vna_averages = vna_averages
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.coarse_sweep_points = coarse_sweep_points
        self.dense_sweep_points = dense_sweep_points
        self.adaptive_sweep_points = adaptive_sweep_points
        self.adaptive_iterations = adaptive_iterations
        self.sg_powers = sg_powers
        self.raw_data_dir = raw_data_dir
        self.plots_dir = plots_dir
        
        self.sg = SG("DEV8488", "localhost")
        self.vna = VNA(vna_ip=vna_ip, vna_timeout_ms=vna_timeout_ms)
        
        self.attenuator = Attenuator(labber_server, labber_server_timeout, attenuator_name, attenuator_interface, attenuator_address)
        self.attenuator_wait_s = attenuator_wait_s
        self.attenuator.set_attenuation(self.readout_power, self.attenuator_wait_s)
        
        self.vna.power_dbm = self.vna_power
        self.vna.ifbw = self.vna_ifbw
        self.vna.averages = self.vna_averages
        
        self.scan_counter = 0

    def vna_data_fetcher(self, new_freqs=None):
        temp_time = time.time()
        freqs, s21_complex, s21_complex_but_off = [], [], []
        
        # 1. Prepare VNA once
        self.vna.prepare_for_cw_sweep(self.readout_frequency)
        
        self.sg.on()
        sweep_freqs = new_freqs if new_freqs is not None else np.linspace(self.start_freq, self.stop_freq, self.dense_sweep_points if self.naive else self.coarse_sweep_points)
        
        for freq in sweep_freqs:
            self.sg.set_cw_tone(freq)
            time.sleep(config.SG_VNA_WAIT_TIME)
            
            # 2. Fast measurement (no mode switching)
            s21 = self.vna.measure_cw_point()
            
            freqs.append(freq)
            s21_complex.append(s21)

        raw_data_filename = f"raw_{self.sg.power_dbm}dBm_dense.npz" if self.naive else f"raw_{self.sg.power_dbm}dBm_coarse.npz"
        self.sg.off()
        
        # 3. Put VNA back to normal
        self.vna.finalize_sweep()


        
        self.vna.prepare_for_cw_sweep(self.readout_frequency)
        
        # self.sg.on()
        sweep_freqs = new_freqs if new_freqs is not None else np.linspace(self.start_freq, self.stop_freq, self.dense_sweep_points if self.naive else self.coarse_sweep_points)
        
        for freq in sweep_freqs:
            self.sg.set_cw_tone(freq)
            time.sleep(config.SG_VNA_WAIT_TIME)
          
            # 2. Fast measurement (no mode switching)
            s21 = self.vna.measure_cw_point()
            
            # freqs.append(freq)
            s21_complex_but_off.append(s21)

        raw_data_filename = f"raw_{self.sg.power_dbm}dBm_dense_off.npz" if self.naive else f"raw_{self.sg.power_dbm}dBm_coarse_off.npz"
        self.sg.off()
        
        # 3. Put VNA back to normal
        self.vna.finalize_sweep()



        print(f"TIME TAKEN: {time.time() - temp_time:.2f}s")
        
        raw_data_path = os.path.join(self.raw_data_dir, raw_data_filename)
        np.savez(raw_data_path, freqs=np.array(freqs), s21=np.array(s21_complex))

        self.scan_counter += 1
        return np.array(freqs), np.array(s21_complex), np.array(s21_complex_but_off)
    
    def vna_data_fetcher_adaptive(self, new_freqs=None):
        freqs, s21_complex = [], []
        self.sg.on()
        
        temp_time = time.time()
        if new_freqs is not None:
            for freq in new_freqs:
                self.sg.set_cw_tone(freq)
                time.sleep(config.SG_VNA_WAIT_TIME)
                s21 = self.vna.measure_cw_point(self.readout_frequency)
                freqs.append(freq)
                s21_complex.append(s21)
            
            raw_data_filename = f"raw_{self.sg.power_dbm}dBm_adaptive_{self.scan_counter}.npz"
        else:
            sweep_freqs = np.linspace(self.start_freq, self.stop_freq, self.dense_sweep_points if self.naive else self.coarse_sweep_points)
            for freq in sweep_freqs:
                self.sg.set_cw_tone(freq)
                time.sleep(config.SG_VNA_WAIT_TIME)
                s21 = self.vna.measure_cw_point(self.readout_frequency)
                freqs.append(freq)
                s21_complex.append(s21)

            raw_data_filename = f"raw_{self.sg.power_dbm}dBm_dense.npz" if self.naive else f"raw_{self.sg.power_dbm}dBm_coarse.npz"
        
        self.sg.off()
        print(f"TIME TAKEN: {time.time() - temp_time:.2}s")
        
        raw_data_path = os.path.join(self.raw_data_dir, raw_data_filename)
        np.savez(raw_data_path, freqs=np.array(freqs), s21=np.array(s21_complex))

        self.scan_counter += 1
        return np.array(freqs), np.array(s21_complex)

    def run(self):
        start = time.time()
        lorentzian = Lorentzian(
            naive=self.naive,
            adaptive_sweep_points=self.adaptive_sweep_points,
            adaptive_iterations=self.adaptive_iterations,
            fetch_data=self.vna_data_fetcher
        )
        
        power_sweep_results = []
        for power in self.sg_powers:
            print(f"--- SG Power {power} dBm ---\n")
            self.sg.set_power(power, config.SG_MAX_ALLOWED_POWER)
            self.scan_counter = 0
            
            f0, kappa, plots_dict, cleaned_data_dict = lorentzian.fitting_routine()
            power_sweep_results.append((power, f0, kappa))
            
            for iteration_name, fig in plots_dict.items():
                plot_filename = f"fitted_{power}dBm_{iteration_name}.png"
                plots_path = os.path.join(self.plots_dir, plot_filename)
                fig.savefig(plots_path, dpi=150)
                plt.close(fig)

        results_array = np.array(power_sweep_results)
        fitted_powers = results_array[:, 0]
        fitted_f0s = results_array[:, 1]
        fitted_kappas = results_array[:, 2]

        baseline_kappa = np.mean(fitted_kappas[:3])
        threshold_kappa = baseline_kappa * config.THRESHOLD_KAPPA_MULTIPLIER
        
        broadened_indices = np.where(fitted_kappas > threshold_kappa)[0]
        
        if len(broadened_indices) > 0:
            optimal_idx = max(0, broadened_indices[0] - 1)
        else:
            optimal_idx = -1 
            
        drive_power = fitted_powers[optimal_idx]
        drive_frequency = fitted_f0s[optimal_idx]
        
        print(f"\n--- OPTIMAL DRIVE FOUND ---")
        print(f"Freq: {drive_frequency/1e9:.5f} GHz | Power: {drive_power} dBm")

        stop = time.time()
        print(f"Time taken: {(stop-start):.2f}s\n")        
        return drive_frequency, drive_power
