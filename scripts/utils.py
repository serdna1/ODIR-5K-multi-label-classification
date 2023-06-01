from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def save_model(model,
               target_dir,
               model_name):
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

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

def split_annotations(annotations_path, target_dir_path, test_ratio = 0.1, val_ratio = 0.2, test_split = True, seed = 42):
    odir_df = pd.read_excel(annotations_path)
    
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    if test_split:
        train_df, test_df = train_test_split(odir_df, test_size = test_ratio, random_state = seed)
        train_df, val_df = train_test_split(train_df, test_size = val_ratio, random_state = seed)
        
        test_df.to_excel(target_dir_path / 'test_annotations.xlsx', index = False)        
    else:    
        train_df, val_df = train_test_split(odir_df, test_size = val_ratio, random_state = seed)
        
    train_df.to_excel(target_dir_path / 'train_annotations.xlsx', index = False)
    val_df.to_excel(target_dir_path / 'val_annotations.xlsx', index = False)