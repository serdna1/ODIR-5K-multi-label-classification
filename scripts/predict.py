from PIL import Image
import torch
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import sys
from utils import load_model

def pred_and_plot_image(model, 
                        left_image_path,
                        right_image_path,
                        transform,
                        device,
                        label_names,
                        ground_truth=None):
    """Makes a prediction on a target image and plots the image with its ground truth and prediction."""
    
    left_img = Image.open(left_image_path)
    right_img = Image.open(right_image_path)
    if transform:
        left_img = transform(left_img)
        right_img = transform(right_img)
    
    model.to(device)
    
    model.eval()
    with torch.inference_mode():
        left_img = left_img.unsqueeze(dim=0).to(device)
        right_img = right_img.unsqueeze(dim=0).to(device)
    
        logits = model(left_img, right_img)
        probs = torch.sigmoid(logits)
        pred_labels = probs>0.5
    
    # MultiLabelBinarazer can convert binary labels in label names and viceversa
    # For example --> possible labels: ['bird','car','plane']; mlb.inverse_transform([[1,0,1]]): [('car', 'plane')]
    mlb = MultiLabelBinarizer(classes=label_names) # The classes atribute is for setting the label order
    mlb.fit([label_names]) # Feed the labels to the object

    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.imshow(left_img.squeeze().permute(1, 2, 0))
    plt.title('Left')
    plt.axis(False)

    plt.subplot(1,2,2)
    plt.imshow(right_img.squeeze().permute(1, 2, 0))
    plt.title('Right')
    plt.axis(False)

    pred_labels = mlb.inverse_transform(pred_labels.cpu().numpy())
    pred_labels = ', '.join(np.squeeze(pred_labels, axis=0))
    if ground_truth:
        ground_truth = mlb.inverse_transform(np.expand_dims(ground_truth, axis=0))
        ground_truth = ', '.join(np.squeeze(ground_truth, axis=0))
        plt.suptitle(f'Ground Truth: {ground_truth} | Pred: {pred_labels}')
    else:
        plt.suptitle(f'Pred: {pred_labels}')

if __name__ == '__main__':
    model_path = sys.argv[1]
    left_image_path = sys.argv[2]
    right_image_path = sys.argv[3]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    pred_and_plot_image(model=load_model(model_path), 
                        left_image_path=left_image_path,
                        right_image_path=right_image_path,
                        transform=transform,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        label_names=['N','D','G','C','A','H','M','O'])   
