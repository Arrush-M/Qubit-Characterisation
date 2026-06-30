import argparse
import config
import data_handler
from experiments.resonator_scan import ResonatorScan
from experiments.qubit_scan import QubitScan


parser = argparse.ArgumentParser()
parser.add_argument("-naive", action="store_true", help="Enable naive mode")
args = parser.parse_args()
NAIVE = args.naive
data_paths = data_handler.generate_data_folder(is_naive=NAIVE)


resonator_scan = ResonatorScan(
    vna_ip=config.VNA_IP, 
    vna_timeout_ms=config.VNA_TIMEOUT_MS,
    vna_start_freq=config.VNA_START_FREQ, 
    vna_stop_freq=config.VNA_STOP_FREQ, 
    vna_center_freq=config.VNA_CENTER_FREQ,
    vna_span_freq=config.VNA_SPAN_FREQ, 
    vna_dense_sweep_points=config.VNA_DENSE_SWEEP_POINTS, 
    vna_coarse_sweep_points=config.VNA_COARSE_SWEEP_POINTS, 
    vna_power_dbm=config.VNA_POWER_DBM, 
    vna_ifbw=config.VNA_IFBW, 
    vna_list_of_averages=config.VNA_LIST_OF_AVERAGES,

    labber_server=config.LABBER_SERVER, 
    labber_server_timeout=config.LABBER_SERVER_TIMEOUT, 
    attenuator_name=config.ATTENUATOR_NAME, 
    attenuator_interface=config.ATTENUATOR_INTERFACE, 
    attenuator_address=config.ATTENUATOR_ADDRESS,
    attenuator_wait_s=config.ATTENUATOR_WAIT_S,

    naive=config.NAIVE, 
    vna_adaptive_sweep_points=config.VNA_ADAPTIVE_SWEEP_POINTS, 
    vna_adaptive_iterations=config.VNA_ADAPTIVE_ITERATIONS,
    attenuator_attenuations=config.ATTENUATOR_ATTENUATIONS,

    s21_dir=data_paths["resonator_scan_s21"],
    plots_dir=data_paths["resonator_scan_plots"],
    logs_path=data_paths["resonator_scan_logs"]
)
READOUT_FREQUENCY, READOUT_ATTENUATION = resonator_scan.run()


# READOUT_FREQUENCY = 6.1627e9
# READOUT_ATTENUATION = 30

qubit_scan = QubitScan(
    naive=config.NAIVE,  
    readout_frequency=READOUT_FREQUENCY,
    readout_power=READOUT_ATTENUATION,
    vna_ip=config.VNA_IP,
    vna_timeout_ms=config.VNA_TIMEOUT_MS,
    vna_power=config.VNA_POWER_DBM,
    vna_ifbw=config.VNA_IFBW, 
    vna_averages=config.VNA_AVERAGES, 
    start_freq=config.SG_START_FREQ, 
    stop_freq=config.SG_STOP_FREQ, 
    coarse_sweep_points=config.SG_COARSE_SWEEP_POINTS, 
    dense_sweep_points=config.SG_DENSE_SWEEP_POINTS, 
    adaptive_sweep_points=config.SG_ADAPTIVE_SWEEP_POINTS, 
    adaptive_iterations=config.SG_ADAPTIVE_ITERATIONS,
    sg_powers=config.SG_POWERS,
    labber_server=config.LABBER_SERVER,
    labber_server_timeout=config.LABBER_SERVER_TIMEOUT,
    sg_name=config.SG_NAME,
    sg_interface=config.SG_INTERFACE,
    sg_address=config.SG_ADDRESS,
    attenuator_name=config.ATTENUATOR_NAME,
    attenuator_interface=config.ATTENUATOR_INTERFACE,
    attenuator_address=config.ATTENUATOR_ADDRESS,
    attenuator_wait_s=config.ATTENUATOR_WAIT_S,
    raw_data_dir=data_paths["qubit_scan_s21"],
    plots_dir=data_paths["qubit_scan_plots"]
)
DRIVE_FREQUENCY, DRIVE_POWER = qubit_scan.run()