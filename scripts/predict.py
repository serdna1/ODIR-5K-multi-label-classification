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
        left_img, right_img = transform([left_img, right_img])
    
    model.to(device)
    
    model.eval()
    with torch.inference_mode():
        left_img = left_img.unsqueeze(dim=0).to(device)
        right_img = right_img.unsqueeze(dim=0).to(device)
    
        logits = model(left_img, right_img)
        probs = torch.sigmoid(logits)
        pred_labels = probs>0.5
    
    # MultiLabelBinarazer can convert binary labels in label names and viceversa
    # For example --> possible labels: ['bird','car','plane']; mlb.inverse_transform([[1,0,1]]): [('bird', 'plane')]
    mlb = MultiLabelBinarizer(classes=label_names) # The classes atribute is for setting the label order
    mlb.fit([label_names]) # Feed the labels to the object

    plt.figure(figsize=(8,4))
    
    plt.subplot(1,2,1)
    plt.imshow(Image.open(left_image_path))
    plt.title('Left')
    plt.axis(False)

    plt.subplot(1,2,2)
    plt.imshow(Image.open(right_image_path))
    plt.title('Right')
    plt.axis(False)

    pred_labels = mlb.inverse_transform(pred_labels.cpu().numpy())
    pred_labels = ', '.join(np.squeeze(pred_labels, axis=0))
    probs = probs.cpu().squeeze().numpy()
    probs = np.around(probs, 2)
    labels_probs_dict = dict(zip(label_names, probs))
    if ground_truth is None:
        plt.suptitle(f'Pred: {pred_labels} | Probs: {labels_probs_dict}')
    else:
        ground_truth = mlb.inverse_transform(np.expand_dims(ground_truth, axis=0))
        ground_truth = ', '.join(np.squeeze(ground_truth, axis=0))
        plt.suptitle(f'Ground Truth: {ground_truth} | Pred: {pred_labels} | Probs: {labels_probs_dict}')

if __name__ == '__main__':
    model_path = sys.argv[1]
    model_name = sys.argv[2]
    left_image_path = sys.argv[3]
    right_image_path = sys.argv[4]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    pred_and_plot_image(model=load_model(model_path, model_name), 
                        left_image_path=left_image_path,
                        right_image_path=right_image_path,
                        transform=transform,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        label_names=['N','D','G','C','A','H','M','O'])   
