import numpy as np
import argparse
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CyclicLR
from torchvision import transforms
import pandas as pd
from datasets import ODIRDataset
from models import create_resnet50_dual
from engine import train
from utils import create_writer, compute_loss_weights
from pytorchtools import EarlyStopping
from samplers import MultilabelBalancedRandomSampler

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
        '--model_name',
        type = str,
        default = 'resnet50_dual',
        help = 'Model arquitecture for training (default: resnet50_dual).'
    )
    parser.add_argument(
        '--images_path',
        type = str,
        default = '/kaggle/input/odir-size-512/odir-size-512',
        help = 'Path of ODIR training images (default: /kaggle/input/odir-size-512/odir-size-512).'
    )
    parser.add_argument(
        '--use_data_augmentation',
        action = 'store_true',
        help = 'If set augment training data.'
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
        '--use_multilabel_balanced_random_sampler',
        action = 'store_true',
        help = 'Use a custom multilabel random sampler that balance the train dataset.'
    )
    parser.add_argument(
        '--num_workers',
        type = int,
        default = os.cpu_count(),
        help = 'Number of workers for data loaders (default: os.cpu_count()).'
    )
    parser.add_argument(
        '--use_weighted_loss',
        action = 'store_true',
        help = 'If set the loss function uses the pos_weight param to consider the dataset imbalance.'
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
        '--lr_scheduler',
        type = str,
        default = None,
        help = 'Learning rate scheduler (default: None)'
    )
    parser.add_argument(
        '--linear_lr_scheduler_total_iters',
        type = int,
        default = 50,
        help = 'Number of steps that the linear lr scheduler decays the learning rate (default: 50).'
    )
    parser.add_argument(
        '--linear_lr_scheduler_start_factor',
        type = int,
        default = 1,
        help = 'Start factor of the linear lr scheduler (default: 1).'
    )
    parser.add_argument(
        '--linear_lr_scheduler_end_factor',
        type = int,
        default = 0.1,
        help = 'End factor of the linear lr scheduler (default: 0.1).'
    )
    parser.add_argument(
        '--cyclic_lr_scheduler_base_lr',
        type = float,
        default = 0.001,
        help = 'Initial learning rate which is the lower boundary in the cycle for each parameter group (default: 0.001).'
    )
    parser.add_argument(
        '--cyclic_lr_scheduler_max_lr',
        type = float,
        default = 0.1,
        help = 'Upper learning rate boundaries in the cycle for each parameter group (default: 0.1).'
    )
    parser.add_argument(
        '--cyclic_lr_scheduler_step_size_up',
        type = int,
        default = 10,
        help = 'Number of training iterations in the increasing half of a cycle (default: 10).'
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

    # Augment data
    if opt.use_data_augmentation:
        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    else:
        augmentations = nn.Identity()
    
    # Create transforms
    train_transform = transforms.Compose([
        augmentations,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if opt.use_normalization else nn.Identity()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if opt.use_normalization else nn.Identity()
    ])

    # Create datasets
    train_df = pd.read_excel(opt.train_annotations_path)
    val_df = pd.read_excel(opt.val_annotations_path)
    
    train_dataset = ODIRDataset(opt.images_path, train_df[:100], train_transform)
    val_dataset = ODIRDataset(opt.images_path, val_df[:30], val_transform)

    if opt.use_multilabel_balanced_random_sampler:
        train_sampler = MultilabelBalancedRandomSampler(
            train_df.loc[:, ['N','D','G','C','A','H','M','O']].to_numpy(),
            list(train_df.index),
            class_choice='least_sampled'
        )
    
    # Create DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=opt.batch_size,
                                  sampler=None if opt.use_multilabel_balanced_random_sampler else train_sampler,
                                  num_workers=opt.num_workers,
                                  pin_memory=True,
                                  shuffle=None if opt.use_multilabel_balanced_random_sampler else True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=opt.batch_size,
                                num_workers=opt.num_workers,
                                pin_memory=True,
                                shuffle=False)
    
    if opt.model_name == 'resnet50_dual':
        model = create_resnet50_dual()
    elif opt.model_name == 'resnet50_dual_v1':
        model = create_resnet50_dual(version=1)
    elif opt.model_name == 'resnet50_dual_v2':
        model = create_resnet50_dual(version=2)

    # Set loss functions
    if opt.use_weighted_loss:
        pos_weight = compute_loss_weights(train_df).to(device)
        train_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        train_loss_fn = torch.nn.BCEWithLogitsLoss()
    val_loss_fn = torch.nn.BCEWithLogitsLoss()

    # Set optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum)
    
    # Set lr scheduler
    if opt.lr_scheduler == 'LinearLR':
        # Reduces lr linearly for a defined number of steps
        scheduler = LinearLR(optimizer,
                             start_factor = opt.linear_lr_scheduler_start_factor,
                             end_factor = opt.linear_lr_scheduler_end_factor,
                             total_iters = opt.linear_lr_scheduler_total_iters)
    elif opt.lr_scheduler == 'CyclicLR':
        scheduler = CyclicLR(optimizer, 
                             base_lr = opt.cyclic_lr_scheduler_base_lr,
                             max_lr = opt.cyclic_lr_scheduler_max_lr,
                             step_size_up = opt.cyclic_lr_scheduler_step_size_up,
                             mode = 'triangular2')
    else:
        scheduler = None
    
    # Initialize the early stopping object
    stopper = EarlyStopping(patience=opt.patience,
                            verbose=True,
                            path=f'./outputs/{opt.model_name}_{opt.experiment_name}_model.pth')
    
    # Create a custom SummaryWriter instance
    writer = create_writer(model_name = opt.model_name,
                           experiment_name = opt.experiment_name,
                           extra = opt.extra)
    
    # Start the training loop
    _, results = train(model=model,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       train_loss_fn=train_loss_fn,
                       val_loss_fn=val_loss_fn,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       epochs=opt.epochs,
                       stopper=stopper,
                       device=device,
                       writer=writer)
    
    # Create a df for the train results and save it to file
    train_results_df = pd.DataFrame(results)
    train_results_df.to_excel('outputs/train_results.xlsx')
    