from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ODIRDataset(Dataset):
    def __init__(self, images_path, annotations_path, indexes, transform=None):
        self.images_path = images_path
        self.df = pd.read_excel(annotations_path)
        self.df = self.df.loc[indexes]
        self.df = self.df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        left_image_path = self.images_path / self.df.at[i, 'Left-Fundus']
        right_image_path = self.images_path / self.df.at[i, 'Right-Fundus']
        left_img = Image.open(left_image_path)
        right_img = Image.open(right_image_path)
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        target = self.df.loc[i, ['A','C','D','G','H','M','N','O']]
        target = target.to_numpy(dtype=np.float32)
        target = torch.tensor(target)
        
        return (left_img, right_img), target

# Usage example
if __name__ == '__main__':
    data_path = Path('./data')
    images_path = data_path / 'ocular-disease-recognition-odir5k' / 'ODIR-5K' / 'ODIR-5K' / 'Training Images'
    annotations_path = data_path / 'ocular-disease-recognition-odir5k' / 'ODIR-5K' / 'ODIR-5K' / 'data.xlsx'

    val_ratio = 0.2
    val_length = round(val_ratio * len(new_df))

    idxs = list(range(len(odir_df)))
    np.random.shuffle(idxs)
    train_idxs = idxs[val_length:]
    val_idxs = idxs[:val_length]

    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = ODIRDataset(images_path, annotations_path, train_idxs, transform)
    val_dataset = ODIRDataset(images_path, annotations_path, val_idxs, transform)
    print(f'Train dataset lenght: {len(train_dataset)}')
    print(f'Validation dataset lenght: {len(val_dataset)}')