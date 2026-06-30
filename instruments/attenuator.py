import time
from qcodes_contrib_drivers.drivers.Vaunix.LDA import Vaunix_LDA
# %%
class Attenuator:
    def __init__(self, labber_server, labber_server_timeout, attenuator_name, attenuator_interface, attenuator_address):
        self.attenuator = Vaunix_LDA(name = attenuator_name, serial_number = 'serial number of the attenuator', dll_path = 'attenuators dll library path')
        
# %%
    def set_attenuation(self, attenuation_db, attenuator_wait_s):
        self.attenuator.attenuation(abs(attenuation_db))
        self.attenuation = attenuation_db
        time.sleep(attenuator_wait_s)
