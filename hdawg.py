import sys
import numpy as np
from zhinst.qcodes import HDAWG

class HDAWG_SG:
    """
    zhinst-qcodes wrapper for Zurich Instruments HDAWG.
    Simplifies the interface to the device.
    """
    def __init__(self, device_serial: str, host: str = "localhost"):
        # 1. Connect to the HDAWG via QCoDeS
        print(f"Connecting to HDAWG {device_serial} on {host}...")
        self.sg = HDAWG(device_serial, host) # not directly "DEV8488" for portability
        
        # 2. Setup AWG to not need LabOne API
        self.sg.awgs[0].set_sequence_params(
            sequence_type="Simple",
            period=20e-6,
            repetitions=1000
        )
        
        # 3. Initialize state variables
        self.freq = None
        self.power_dbm = None
        self.phase = 0.0 # mostly relevant for CW

    @staticmethod
    def dbm_to_vp(dbm: float) -> float:
        """Converts dBm to amplitude assuming a standard 50 ohm load."""
        return np.sqrt(0.1 * 10**(dbm / 10))

    # I would keep set_cw_tone and set_power as part of prepare/send waveform, but for legacy (sg.py) reasons kept separate
    def set_cw_tone(self, freq: float): 
        """Sets the Continuous Wave frequency."""
        self.freq = freq
        print(f"--- HDAWG Frequency set to {self.freq/1e9:.4f} GHz")
        
    def set_power(self, power_dbm: float, max_power: float = 0): 
        """Sets the power level property and validates against safety limits."""
        if power_dbm >= max_power: 
            print(f"SAFETY TRIGGERED: Power {power_dbm} dBm is >= {max_power} dBm. Aborting.")
            sys.exit()
            
        self.power_dbm = power_dbm
        print(f"--- HDAWG Power set to {self.power_dbm} dBm")

    def prepare_cw_waveform(self):
        """Internal helper to compile and push the array memory into the HDAWG."""
        if self.freq is None or self.power_dbm is None:
            raise ValueError("Frequency and Power must be set before generating CW waveform.")

        # Rescale wave to desired power    
        target_vp = self.dbm_to_vp(self.power_dbm)
        self.sg.sigouts[0].range(target_vp)
        actual_range = self.sg.sigouts[0].range()
        
        # Re-initialize the HDAWG's QCoDeS internal array queue
        self.sg.awgs[0].reset_queue()
        
        t = np.linspace(0, 20e-6, 8000)
        w = np.cos(2 * np.pi * self.freq * t + self.phase) * (target_vp / actual_range)
        
        # Upload to AWG queue and compile
        self.sg.awgs[0].queue_waveform(w, [])
        self.sg.awgs[0].compile_and_upload_waveforms()

    def prepare_arbitrary_waveforms(self, wave1: np.ndarray, wave2: np.ndarray, power_dbm: float, max_power: float = 0):
        """
        Standalone method ported from procedural script to upload custom arbitrary waveforms.
        Future-proofing for time-domain/qubit measurements.
        """
        # For single complex waveform, HDAWG automatically splits into channels 1 and 2. 
        # wave1=[complex], wave2=[] will simplify to wave1=[Re(complex)], wave2=[Im(complex)]. 

        if power_dbm >= max_power: 
            print(f"SAFETY TRIGGERED: Power {power_dbm} dBm is >= {max_power} dBm. Aborting.")
            sys.exit()

        target_vp = self.dbm_to_vp(power_dbm)
        self.sg.sigouts[0].range(target_vp)
        actual_range = self.sg.sigouts[0].range()
        
        self.sg.awgs[0].reset_queue()
        
        # Scale to set the correct amplitude
        w1_scaled = wave1 * (target_vp / actual_range)
        w2_scaled = wave2 * (target_vp / actual_range)
        
        self.sg.awgs[0].queue_waveform(w1_scaled, w2_scaled)
        self.sg.awgs[0].compile_and_upload_waveforms()
        
        print("\n--- HDAWG Custom Waveform Output: ON")

    def on(self):
        """Toggles hardware output."""
        self.sg.sigouts[0].on(True)
        self.sg.awgs[0].enable(True)
        print("\n--- HDAWG Output: ON")

    def off(self):
        """Stops AWG sequencer loop and disables physical outputs."""
        self.sg.sigouts[0].on(False)
        self.sg.awgs[0].enable(False)
        print("\n--- HDAWG Output: OFF")
        
    