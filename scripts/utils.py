from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def create_writer(experiment_name, 
                  model_name, 
                  extra):
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    
    return SummaryWriter(log_dir=log_dir)