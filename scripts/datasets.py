from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class ODIRDataset(Dataset):
    def __init__(self, images_path, df, transform=None, join_images=False):
        self.images_path = images_path
        self.df = df
        self.transform = transform
        self.join_images = join_images
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        left_image_path = f'{self.images_path}/{self.df.at[i, "Left-Fundus"]}'
        right_image_path = f'{self.images_path}/{self.df.at[i, "Right-Fundus"]}'
        left_img = Image.open(left_image_path)
        right_img = Image.open(right_image_path)
        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        labels = ['A','C','D','G','H','M','N','O']
        target = self.df.loc[i, labels]
        target = target.to_numpy(dtype=np.float32)
        target = torch.tensor(target)
        
        if self.join_images:
            pair_img = torch.cat((left_img, right_img), dim=2)
            return pair_img, target
        
        return (left_img, right_img), target
