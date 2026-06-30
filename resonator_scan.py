import os
import time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from instruments.vna import VNA
from instruments.attenuator import Attenuator
from fitting_algorithms.lorentzian import Lorentzian


class ResonatorScan:
    def __init__(self, vna_ip, vna_timeout_ms, vna_start_freq, vna_stop_freq, vna_center_freq, vna_span_freq, vna_dense_sweep_points, vna_coarse_sweep_points, vna_power_dbm, vna_ifbw, vna_list_of_averages, labber_server, labber_server_timeout, attenuator_name, attenuator_interface, attenuator_address, attenuator_wait_s, naive, vna_adaptive_sweep_points, vna_adaptive_iterations, attenuator_attenuations, s21_dir, plots_dir, logs_path):
        self.vna = VNA(vna_ip=vna_ip, vna_timeout_ms=vna_timeout_ms)
        self.vna_start_freq = vna_start_freq
        self.vna_stop_freq = vna_stop_freq
        self.vna_center_freq = vna_center_freq
        self.vna_span_freq = vna_span_freq
        self.vna_coarse_sweep_points = vna_coarse_sweep_points
        self.vna_dense_sweep_points = vna_dense_sweep_points
        self.vna.power_dbm = vna_power_dbm
        self.vna.ifbw = vna_ifbw
        self.list_of_averages = vna_list_of_averages
        
        self.attenuator = Attenuator(labber_server=labber_server, labber_server_timeout=labber_server_timeout, attenuator_name=attenuator_name, attenuator_interface=attenuator_interface, attenuator_address=attenuator_address)
        self.attenuator_wait_s = attenuator_wait_s
        self.attenuator_attenuations = attenuator_attenuations
        
        self.naive = naive
        self.vna_adaptive_sweep_points = vna_adaptive_sweep_points
        self.vna_adaptive_iterations = vna_adaptive_iterations
        self.attenuator_attenuations = attenuator_attenuations

        self.s21_dir = s21_dir
        self.plots_dir = plots_dir
        self.logs_path = logs_path

        self.scan_counter = None

    def vna_data_fetcher(self, new_freqs=None):
        if new_freqs is not None:
            self.vna.freqs = new_freqs
            raw_data_filename = f"raw_{self.attenuator.attenuation}dB_adaptive_{self.scan_counter}.npz"
            plot_filename = f"vna_{self.attenuator.attenuation}dB_adaptive_{self.scan_counter}.png"
            freqs, s21_complex = self.vna.scan(custom_freqs=True)
        else:
            self.vna.set_sweep(start_freq=self.vna_start_freq, 
                               stop_freq=self.vna_stop_freq, 
                               center_freq=self.vna_center_freq, 
                               span_freq=self.vna_span_freq, 
                               sweep_points=self.vna_dense_sweep_points if self.naive else self.vna_coarse_sweep_points)
            raw_data_filename = f"raw_{self.attenuator.attenuation}dB_dense.npz" if self.naive else f"raw_{self.attenuator.attenuation}dB_coarse.npz"
            plot_filename = f"vna_{self.attenuator.attenuation}dB_dense.png" if self.naive else f"vna_{self.attenuator.attenuation}dB_coarse.png"
            freqs, s21_complex = self.vna.scan()

        raw_data_path = os.path.join(self.s21_dir, raw_data_filename)
        np.savez(raw_data_path, freqs=freqs, s21=s21_complex)
        
        plots_path = os.path.join(self.plots_dir, plot_filename)
        fig_vna = self.vna.plot()
        fig_vna.savefig(plots_path, dpi=150)
        plt.close(fig_vna)

        self.scan_counter += 1
        return freqs, s21_complex

    def run(self):
        exp_start_time = time.time()
        fitter = Lorentzian(
            naive=self.naive,
            adaptive_sweep_points=self.vna_adaptive_sweep_points,
            adaptive_iterations=self.vna_adaptive_iterations,
            fetch_data=self.vna_data_fetcher
        )
        
        with open(self.logs_path, "a") as log: 
            log.write(f"--- RESONATOR SCAN LOG ---\n")

        attenuation_sweep_results = []
        try:
            for attenuation in self.attenuator_attenuations:
                loop_start_time = time.time()
                
                self.vna.averages = self.list_of_averages[self.attenuator_attenuations.index(attenuation)]
                self.attenuator.set_attenuation(attenuation, self.attenuator_wait_s)
                print(f"--- Attenuator Attenuation {attenuation} dB ---\n")

                self.scan_counter = 0 
                f0, kappa, plots_dict, cleaned_data_dict = fitter.fitting_routine()
                attenuation_sweep_results.append((attenuation, f0, kappa))
                
                for iteration_name, fig in plots_dict.items():
                    plot_filename = f"fitted_{attenuation}dB_{iteration_name}.png" 
                    plots_path = os.path.join(self.plots_dir, plot_filename)
                    fig.savefig(plots_path, dpi=150)
                    plt.close(fig)            
                
                for iteration_name, data_arrays in cleaned_data_dict.items():
                    data_filename = f"cleaned_{attenuation}dB_{iteration_name}.npz"
                    data_path = os.path.join(self.s21_dir, data_filename)
                    np.savez(data_path, **data_arrays)

                loop_time = time.time() - loop_start_time
                with open(self.logs_path, "a") as log:
                    log.write(f"Attenuation: {attenuation:>4} dB | f0: {f0/1e9:.6f} GHz | kappa: {kappa/1e6:.4f} MHz | Time: {loop_time:.2f}s\n")
        finally:
            self.vna.reset()
    

        results_array = np.array(attenuation_sweep_results)
        fitted_attenuations = results_array[:, 0]
        fitted_f0s = results_array[:, 1]
        print(f"Attenuations: {fitted_attenuations}")
        print(f"Resonance Frequencies: {fitted_f0s}")

        baseline_f0 = np.mean(fitted_f0s[:3])
        shift_tolerance = 50e3 
        
        deviations = np.abs(fitted_f0s - baseline_f0)
        crash_indices = np.where(deviations > shift_tolerance)[0]
        
        if len(crash_indices) > 0:
            optimal_idx = crash_indices[0] - 1
            optimal_idx = max(0, optimal_idx)
        else:
            optimal_idx = -1 
            
        readout_attenuation = fitted_attenuations[optimal_idx]
        readout_frequency = fitted_f0s[optimal_idx]
        
        print(f"\n--- OPTIMAL READOUT FOUND ---")
        print(f"Freq: {readout_frequency/1e9:.5f} GHz | Attenuation: {readout_attenuation} dB")

        exp_stop_time = time.time()
        print(f"Time taken: {(exp_stop_time-exp_start_time):.2f}s\n")   

        with open(self.logs_path, "a") as log:
            log.write(f"\n--- OPTIMAL READOUT FOUND ---\n")
            log.write(f"Freq: {readout_frequency/1e9:.5f} GHz | Attenuation: {readout_attenuation} dBm\n")
            log.write(f"Total Experiment Time: {(exp_stop_time-exp_start_time):.2f}s\n")
     
        return readout_frequency, readout_attenuation
