from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ODIRDataset(Dataset):
    def __init__(self, images_path, df, transform=None, join_images=False):
        self.images_path = images_path
        self.df = df
        self.transform = transform
        self.join_images = join_images
        
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
        
        if self.join_images:
            pair_img = torch.cat((left_img, right_img), dim=2)
            return pair_img, target
        
        return (left_img, right_img), target

# Usage example
if __name__ == '__main__':
    images_path = Path('data/images')
    annotations_path = Path('data/annotations.xlsx')

    transform = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor(),
    ])
    
    odir_df = pd.read_excel(annotations_path)

    test_ratio = 0.1
    val_ratio = 0.2

    label_names = ['A','C','D','G','H','M','N','O']
    
    train_df, test_df = train_test_split(odir_df, test_size=test_ratio, random_state=42, stratify=odir_df.loc[:, label_names])
    train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=42, stratify=train_df.loc[:, label_names])

    train_dataset = ODIRDataset(images_path, train_df, transform, join_images=True)
    val_dataset = ODIRDataset(images_path, val_df, transform, join_images=True)
    test_dataset = ODIRDataset(images_path, test_df, transform, join_images=True)

    print(f'Train dataset lenght: {len(train_dataset)}')
    print(f'Validation dataset lenght: {len(val_dataset)}')
    print(f'Test dataset lenght: {len(test_dataset)}')