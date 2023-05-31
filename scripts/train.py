import numpy as np
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datasets import ODIRDataset
from engine import train
from utils import save_model, create_writer

def get_args_parser():
    parser = argparse.ArgumentParser(
        description = 'Train a network for image multi-label classification on ODIR-5K.'
    )
    parser.add_argument(
        '--random_seed',
        type = int,
        default = 42,
        help = 'Sets random seed for reproducibility (default: 42).'
    )
    parser.add_argument(
        '--images_path',
        type = str,
        default = '/kaggle/input/odir-size-512/odir-size-512',
        help = 'Path of ODIR training images (default: /kaggle/input/odir-size-512/odir-size-512).'
    )
    parser.add_argument(
        '--train_annotations_path',
        type = str,
        default = '/kaggle/input/odir-size-512/train_annotations.xlsx',
        help = 'Path of train annotations file (default: /kaggle/input/odir-size-512/train_annotations.xlsx).'
    )
    parser.add_argument(
        '--val_annotations_path',
        type = str,
        default = '/kaggle/input/odir-size-512/val_annotations.xlsx',
        help = 'Path of validation annotations file (default: /kaggle/input/odir-size-512/val_annotations.xlsx).'
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 32,
        help = 'Batch size for data loaders (default: 32).'
    )
    parser.add_argument(
        '--num_workers',
        type = int,
        default = os.cpu_count(),
        help = 'Number of workers for data loaders (default: os.cpu_count()).'
    )
    parser.add_argument(
        '--lr',
        type = float,
        default = 1e-3,
        help = 'Specifies learning rate for optimizer (default: 1e-3).'
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 5,
        help = 'Number of training epochs (default: 5).'
    )
    parser.add_argument(
        '--experiment_name',
        type = str,
        default = 'experiment_0',
        help = 'Name of the experiment" (default: experiment_0).'
    )
    parser.add_argument(
        '--extra',
        type = str,
        default = None,
        help = 'Extra info about the experiment (default: None).'
    )

    return parser

if __name__ == '__main__':
    opt = get_args_parser().parse_args()

    # Reproducibility
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    transform = transforms.ToTensor()

    # Create datasets
    train_df = pd.read_excel(opt.train_annotations_path)
    val_df = pd.read_excel(opt.val_annotations_path)
    
    train_dataset = ODIRDataset(opt.images_path, train_df, transform, join_images=True)
    val_dataset = ODIRDataset(opt.images_path, val_df, transform, join_images=True)

    # Create DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  pin_memory=True,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                shuffle=False)
    
    # Import a resnet50 from pytorch
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Recreate the fc layer and seed it to the target device
    input_shape = model.fc.in_features
    output_shape = 8 # number of labels
    model.fc = torch.nn.Linear(input_shape, output_shape).to(device)

    # Set loss function and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr)
    
    # Start the training loop
    results = train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    epochs=opt.epochs,
                    device=device,
                    writer=create_writer(experiment_name = opt.experiment_name,
                                         model_name = 'resnet50',
                                         extra = opt.extra))

    # Save the model
    save_model(model=model,
               target_dir="models",
               model_name="model.pth")