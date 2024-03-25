from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class ODIRDataset(Dataset):
    def __init__(self, images_path, df, transform=None):
        self.images_path = images_path
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        left_image_path = f'{self.images_path}/{self.df.at[i, "Left-Fundus"]}'
        right_image_path = f'{self.images_path}/{self.df.at[i, "Right-Fundus"]}'
        left_img = Image.open(left_image_path)
        right_img = Image.open(right_image_path)
        if self.transform:
            left_img, right_img = self.transform([left_img, right_img])
        
        target = self.df.loc[i, ['N','D','G','C','A','H','M','O']]
        target = target.to_numpy(dtype=np.float32)
        target = torch.tensor(target)
        
        return (left_img, right_img), target
