import numpy as np
import qcodes as qc
import matplotlib.pyplot as plt
import sys
from qcodes.instrument_drivers.rohde_schwarz import RohdeSchwarzZNB20
from qcodes.dataset import (
    Measurement,
    initialise_database,
    load_or_create_experiment,
    plot_by_id,
    ArraySweep
)
# %%


class VNA:
    def __init__(self, vna_ip, vna_timeout_ms, channel=1):
        self.channel = channel
        self.start_freq = None
        self.stop_freq = None
        self.center_freq = None
        self.span_freq = None
        self.sweep_points = None
        self.freqs = None
        
        self.power_dbm = None
        self.ifbw = None
        self.averages = None
        
        self.s21_complex = None
# %%
        try:
            self.vna = RohdeSchwarzZNB20(name='vna_name', address = vna_ip, timeout = vna_timeout_ms, terminator = '\n')
        except Exception as e:
            print(f"Failed to connect to VNA: {e}")
            sys.exit()
# %%
    def set_sweep(self, start_freq, stop_freq, center_freq, span_freq, sweep_points):
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.center_freq = center_freq
        self.span_freq = span_freq
        self.sweep_points = sweep_points
        
        if start_freq is None or stop_freq is None:
            start_freq = center_freq - (span_freq/2)
            stop_freq = center_freq + (span_freq/2)
# %%
    def reset(self):
        self.vna.cont_meas_on()
        self.vna.write("*CLS")
# %%
    def scan(self, custom_freqs=False):
        if not custom_freqs:
            print("--- [VNA Scan: Standard Linear Sweep] ---")
            print(f"  Range: {self.start_freq/1e9:.4f} - {self.stop_freq/1e9:.4f} GHz")
            print(f"  Power: {self.power_dbm} dBm | IFBW: {self.ifbw} Hz | Avg: {self.averages}")
            
            self.vna.write("*CLS")
            self.vna.channels.S21.power(self.power_dbm)
            self.vna.channels.S21.start(self.start_freq)
            self.vna.channels.S21.stop(self.stop_freq)
            self.vna.channels.S21.npts(self.sweep_points)
            self.vna.channels.S21.bandwidth(self.ifbw)
                        
            self.vna.cont_meas_off()
            self.vna.channels.S21.avg(self.averages)
            
            self.vna.rf_on()
            self.vna.channels.autoscale()
            
            raw = self.vna.channels.S21.trace_mag_phase.get()
            data_complex = raw[0] + 1j * raw[1]
            self.s21_complex = data_complex
            
        else:
            print("--- [VNA Scan: Custom Point-by-Point Sweep] ---")
            print(f"  Total Points: {self.sweep_points}")
            print(f"  Power: {self.power_dbm} dBm | IFBW: {self.ifbw} Hz | Avg: {self.averages}")
            
            self.vna.write("*CLS")
            self.vna.channels.S21.power(self.power_dbm)
            self.vna.channels.S21.bandwidth(self.ifbw)
            
            self.vna.cont_meas_off()
            self.vna.channels.S21.avg(self.averages)
            self.vna.rf_on()
            
            raw = ArraySweep(self.vna.channels.S21.trace_mag_phase, self.freqs)
            adaptive_s21 = raw[0] + 1j * raw[1]
            self.s21_complex = adaptive_s21
            
            self.vna.channels.S21.start(self.start_freq)
            self.vna.channels.S21.stop(self.stop_freq)
            self.vna.channels.autoscale()
            
        print("--- [Hardware Scan Complete] ---\n")
        return self.freqs, self.s21_complex
# %%
    def prepare_for_cw_sweep(self, readout_freq_hz):
        self.vna.write("*CLS")
        self.vna.channels.S21.power(self.power_dbm)
        self.vna.channels.S21.setup_cw_sweep()
        self.vna.channels.S21.cw_frequency(readout_freq_hz)
        self.vna.channels.S21.bandwidth(self.ifbw)
        self.vna.channels.S21.update_cw_traces()
        
        self.vna.channels.S21.avg(self.averages)
        
        self.vna.cont_meas_off()
        self.vna.rf_on()
# %%
    def measure_cw_point(self):
        raw = self.vna.channels.S21.trace_mag_phase.get()
        return raw[0] + 1j * raw[1]
# %%
    def finalize_sweep(self):
        self.vna.channels.S21.setup_lin_sweep()
        self.vna.cont_meas_on()
# %%
    def plot(self):
        s21 = self.s21_complex
        s21_mag = np.abs(s21)
        s21_phase = np.angle(s21)

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Hardware VNA Scan: P={self.power_dbm} dBm | Avg={self.averages}", fontsize=14)

        plt.subplot(1, 3, 1)
        plt.plot(self.freqs, s21_mag, "b")
        plt.title("Magnitude")
        plt.ylabel("|S21|"); plt.xlabel("Frequency (GHz)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(self.freqs, s21_phase, "r")
        plt.title("Phase")
        plt.ylabel("Radians"); plt.xlabel("Frequency (GHz)")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        plt.plot(s21.real, s21.imag, "g-", alpha=0.5, marker="o", markersize=3)
        
        res_idx = np.argmin(np.abs(s21)) 
        plt.plot(s21.real[res_idx], s21.imag[res_idx], "ro", label="Res")
        plt.plot(s21.real[0], s21.imag[0], "kx", label="Start")
        
        plt.title("IQ Circle")
        plt.xlabel("I"); plt.ylabel("Q")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
