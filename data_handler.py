import os
from datetime import datetime
import shutil


def generate_data_folder(is_naive):
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    
    suffix = "_naive" if is_naive else ""
    base_dir = os.path.join("data", run_timestamp + suffix)

    paths = {
        "resonator_scan_s21":  os.path.join(base_dir, "resonator_scan", "s21"),
        "resonator_scan_plots": os.path.join(base_dir, "resonator_scan", "plots"),
        "resonator_scan_logs": os.path.join(base_dir, "resonator_scan", "logs.txt"),
        
        "qubit_scan_s21":   os.path.join(base_dir, "qubit_scan", "s21"),
        "qubit_scan_plots":  os.path.join(base_dir, "qubit_scan", "plots"),
        "qubit_scan_logs": os.path.join(base_dir, "qubit_scan", "logs.txt"),
    }

    for path in paths.values():
        if not path.endswith(".txt"):
            os.makedirs(path, exist_ok=True)

    shutil.copy("config.py", os.path.join(base_dir, "config_snapshot.txt"))
    
    return paths
