from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from models import create_resnet50_dual
from PIL import Image
from tqdm.auto import tqdm

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
  
def load_model(model_path):
    model = create_resnet50_dual()

    model.load_state_dict(torch.load(model_path))

    return model

def create_writer(model_name,
                  experiment_name,  
                  extra):
    if extra:
        log_dir = os.path.join('..', 'runs', model_name, experiment_name, extra)
    else:
        log_dir = os.path.join('..', 'runs', model_name, experiment_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    
    return SummaryWriter(log_dir=log_dir)

def split_annotations(annotations_path, target_dir_path, test_ratio = 0.1, val_ratio = 0.2, test_split = True, seed = 42):
    odir_df = pd.read_excel(annotations_path)
    
    Path(target_dir_path).mkdir(parents=True, exist_ok=True)
    
    if test_split:
        train_df, test_df = train_test_split(odir_df, test_size = test_ratio, random_state = seed)
        train_df, val_df = train_test_split(train_df, test_size = val_ratio, random_state = seed)
        
        test_df.to_excel(f'{target_dir_path}/test_annotations.xlsx', index = False)        
    else:    
        train_df, val_df = train_test_split(odir_df, test_size = val_ratio, random_state = seed)
        
    train_df.to_excel(f'{target_dir_path}/train_annotations.xlsx', index = False)
    val_df.to_excel(f'{target_dir_path}/val_annotations.xlsx', index = False)

def create_annotations_mini(source_path,
                            target_path,
                            labels = ['N','D','G','C','A','H','M','O'],
                            images_per_label = 9999,
                            seed = 42):
    df = pd.read_excel(source_path)
    df = df.loc[:, ['ID','Left-Fundus','Right-Fundus'] + labels]
    
    mini_df = pd.DataFrame()
    for label in labels:
        n_samples = min( int(df.loc[:, [label]].sum()), images_per_label )
        label_sample_df = df[(df[label] > 0)].sample(n_samples, random_state=seed)
        df = df.drop(label_sample_df.index)
        mini_df = pd.concat([mini_df, label_sample_df])
    mini_df = mini_df.reset_index(drop=True)
    
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    mini_df.to_excel(target_path)
    
    return mini_df

def preprocess_images(data_dir,
                      transform,
                      target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    image_path_list = list(Path(data_dir).glob("*.jpg"))
    for image_path in tqdm(image_path_list):
        img = Image.open(image_path)
        img = transform(img)
        save_path = f'{target_dir}/{image_path.name}'
        img.save(save_path)