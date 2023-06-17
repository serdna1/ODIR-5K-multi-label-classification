import numpy as np
import argparse
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from datasets import ODIRDataset
from models import create_resnet50_dual
from engine import train
from utils import create_writer
from pytorchtools import EarlyStopping

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
        '--use_normalization',
        action = 'store_true',
        help = 'If set the images are normalized the same way the imagenet images where normalized to train the resnet50 used here.'
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
        '--momentum',
        type = float,
        default = 0.9,
        help = 'Specifies momentum for optimizer (default: 0.9).'
    )
    parser.add_argument(
        '--use_lr_scheduler',
        action = 'store_true',
        help = 'If set a lr scheduler is used.'
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 5,
        help = 'Number of training epochs (default: 5).'
    )
    parser.add_argument(
        '--patience',
        type = int,
        default = 1,
        help = 'How long to wait after last time validation loss improved (default: 1).'
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

    # Create the outputs directory
    Path('outputs/').mkdir(parents=True, exist_ok=True)
    
    # Reproducibility
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed(opt.random_seed)

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if opt.use_normalization else nn.Identity()
    ])

    # Create datasets
    train_df = pd.read_excel(opt.train_annotations_path)
    val_df = pd.read_excel(opt.val_annotations_path)
    
    train_dataset = ODIRDataset(opt.images_path, train_df, transform)
    val_dataset = ODIRDataset(opt.images_path, val_df, transform)

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
    
    model = create_resnet50_dual()

    # Set loss function and optimizer
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum)
    
    if opt.use_lr_scheduler:
        # Decay lr by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None
    
    # Initialize the early stopping object
    stopper = EarlyStopping(patience=opt.patience,
                            verbose=True,
                            path=f'./outputs/resnet50_dual_{opt.experiment_name}_model.pth')
    
    # Create a custom SummaryWriter instance
    writer = create_writer(model_name = 'resnet50_dual',
                           experiment_name = opt.experiment_name,
                           extra = opt.extra)
    
    # Start the training loop
    _, results = train(model=model,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       loss_fn=loss_fn,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epochs=opt.epochs,
                       stopper=stopper,
                       device=device,
                       writer=writer)
    
    # Create a df for the train results and save it to file
    train_results_df = pd.DataFrame(results)
    train_results_df.to_excel('outputs/train_results.xlsx')
    