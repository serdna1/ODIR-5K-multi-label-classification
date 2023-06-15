import argparse
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from datasets import ODIRDataset
from utils import load_model
from engine import val_step

def get_args_parser():
    parser = argparse.ArgumentParser(
        description = 'Test a model with a dataset.'
    )
    parser.add_argument(
        '--model_path',
        type = str,
        default = 'outputs/resnet50_dual_experiment_0_model.pth',
        help = 'Path to the model to test the dataset with (default: outputs/resnet50_dual_experiment_0_model.pth).'
    )
    parser.add_argument(
        '--images_path',
        type = str,
        default = '../../data/train_fov_cc_fov_224/',
        help = 'Path of test images (default: ../../data/train_fov_cc_fov_224/).'
    )
    parser.add_argument(
        '--test_annotations_path',
        type = str,
        default = 'data/test_annotations.xlsx',
        help = 'Path of test annotations file (default: data/test_annotations.xlsx).'
    )
    parser.add_argument(
        '--use_normalization',
        action = 'store_true',
        help = 'If set the images are normalized the same way the imagenet images were normalized to train the resnet50 used here.'
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
        '--ground_truth_path',
        type = str,
        default = None,
        help = 'Path to store the ground truth labels used in test (default: None).'
    )
    parser.add_argument(
        '--probs_path',
        type = str,
        default = None,
        help = 'Path to store the result probs of test (default: None).'
    )

    return parser

if __name__ == '__main__':
    opt = get_args_parser().parse_args()

    # Create the outputs directory if not exists already
    Path('outputs/').mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(opt.model_path)
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if opt.use_normalization else nn.Identity() # nn.Identity is like not using a transform
    ])

    # Create test dataset
    test_df = pd.read_excel(opt.test_annotations_path)
    test_dataset = ODIRDataset(opt.images_path, test_df, transform)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.num_workers,
                                 pin_memory=True,
                                 shuffle=False)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Test the model
    (loss,
    kappa,
    f1,
    auc,
    final_score,
    all_y,
    all_probs) = val_step(model, 
                          test_dataloader, 
                          loss_fn,
                          device)
    
    print(
        f"test_loss: {loss:.4f} | "
        f"test_kappa: {kappa:.4f} | "
        f"test_f1: {f1:.4f} | "
        f"test_auc: {auc:.4f} | "
        f"test_final: {final_score:.4f}\n"
    )
    
    if opt.ground_truth_path:
        # Create a ground truth df and save it to a file
        gt_df = test_df.loc[:, ['ID','N','D','G','C','A','H','M','O']]
        gt_df = gt_df.reset_index(drop=True)
        gt_df.to_excel(opt.ground_truth_path, index=False)

    if opt.probs_path:
        # Create a df to store the probabilities of each patient of the test dataset
        probs_df = pd.DataFrame(all_probs, columns=['A','C','D','G','H','M','N','O'])
        probs_df.insert(0, 'ID', test_df['ID'])
        probs_df = probs_df.loc[:, ['ID','N','D','G','C','A','H','M','O']]
        probs_df.to_excel(opt.probs_path, index=False)
    