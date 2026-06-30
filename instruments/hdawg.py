import sys
import numpy as np
from zhinst.qcodes import HDAWG
from zhinst.toolkit import Waveforms

class HDAWG:
    """
    zhinst-qcodes wrapper for Zurich Instruments HDAWG.
    Simplifies the interface to the device.
    """
    def __init__(self, device_serial: str, host: str = "localhost"):

        # Connect to the HDAWG via QCoDeS

        print(f"Connecting to HDAWG {device_serial} on {host}...")
        
        self.sg = HDAWG(device_serial, host) # not directly "DEV8488" for portability
        
        # Initialize state variables
        self.freq = None
        self.power_dbm = None
        self.phase = 0.0 # mostly relevant for CW
        self._compiled_wfm_len = 0
        self._compiled_dual = False

    @staticmethod
    def dbm_to_vp(dbm: float) -> float:
        """Converts dBm to amplitude assuming a standard 50 ohm load."""
        return np.sqrt(0.1 * 10**(dbm / 10))

    # I would keep set_cw_tone and set_power as part of prepare/send waveform, but for legacy (sg.py) reasons kept separate
    def set_cw_tone(self, freq: float): 
        self.freq = freq
        print(f"--- HDAWG Frequency set to {self.freq/1e9:.4f} GHz")
        
    def set_power(self, power_dbm: float, max_power: float = 0): 
        """Sets the power level property and validates against safety limits."""
        if power_dbm >= max_power: 
            print(f"SAFETY TRIGGERED: Power {power_dbm} dBm is >= {max_power} dBm. Aborting.")
            sys.exit()
            
        self.power_dbm = power_dbm
        print(f"--- HDAWG Power set to {self.power_dbm} dBm")

    def _ensure_sequencer_compiled(self, length: int, dual: bool = False):

        # Bypasses the slow compilation phase if waveform remains same, as advised in zhinst-toolkit documentation

        if self._compiled_wfm_len == length and self._compiled_dual == dual:
            return 

        print(f"[{self.sg.name}] Compiling AWG FPGA code (This happens rarely)...")
        
        if not dual:
            seqc_program = f"""
            wave w1 = placeholder({length});      // Reserve memory for a 1-Ch shape
            assignWaveIndex(1, w1, 0);          // Link w1 to memory slot 0
            while (true) {{ playWave(1, w1); }}    // Replaces repetitions/infinite loop
            """ # 1 channel if dual=False
        else:
            seqc_program = f"""
            wave w = placeholder({length}, true, false);
            assignWaveIndex(1, 2, w, 0);                
            while (true) {{ playWave(1, 2, w); }}
            """ # 2 channels if dual=True

        self.sg.awgs[0].load_sequencer_program(seqc_program)
        
        # Update cache tracking
        self._compiled_wfm_len = length
        self._compiled_dual = dual

    def _prepare_cw_waveform(self):
        if self.freq is None or self.power_dbm is None:
            raise ValueError("Frequency and Power must be set before generating CW waveform.")
        
        # Rescaling constants 
        target_vp = self.dbm_to_vp(self.power_dbm)
        self.sg.sigouts[0].range(target_vp)
        actual_range = self.sg.sigouts[0].range()
        
        wfm_length = 8000
        t = np.linspace(0, 20e-6, wfm_length)

        w_scaled = np.sin(2 * np.pi * self.freq * t + self.phase) * (target_vp / actual_range)

        self._ensure_sequencer_compiled(wfm_length, dual=False)

        waveforms = Waveforms()
        waveforms.assign_waveform(0, w_scaled)  # Push w_scaled to hardware memory slot 0
        # wave2 argument not required here
        self.sg.awgs[0].write_to_waveform_memory(waveforms)

    def _prepare_arbitrary_waveforms(self, wave1: str, wave2: str, power_dbm: float, max_power: float = 0):
        """
        Future-proofing for time-domain/qubit measurements.
        """
        # For single complex waveform, HDAWG should automatically split into channels 1 and 2. 
        # e.g wave1=[complex], wave2=[] will simplify to wave1=[Re(complex)], wave2=[Im(complex)]. 

        if power_dbm >= max_power: 
            print(f"SAFETY TRIGGERED: Power {power_dbm} dBm is >= {max_power} dBm. Aborting.")
            sys.exit()

        target_vp = self.dbm_to_vp(power_dbm)
        self.sg.sigouts[0].range(target_vp)
        actual_range = self.sg.sigouts[0].range()
        
        if len(wave1) != len(wave2):
            raise ValueError("Lengths of channel 1 and channel 2 waves must be identical.")
        
        self._ensure_sequencer_compiled(len(wave1), dual=True)
        
        scaled_w1 = wave1 * (target_vp / actual_range)
        scaled_w2 = wave2 * (target_vp / actual_range)
        
        waveforms = Waveforms()
        waveforms.assign_waveform(0, scaled_w1, scaled_w2)
        
        self.sg.awgs[0].write_to_waveform_memory(waveforms)
        
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
        
    
    
