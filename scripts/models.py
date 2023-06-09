import torchvision
import torch

def create_resnet50(device):
    # Import a resnet50 from pytorch
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).to(device)

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Recreate the fc layer and seed it to the target device
    input_shape = model.fc.in_features
    output_shape =  8 # number of labels
    model.fc = torch.nn.Linear(input_shape, output_shape).to(device)

    # Give the model a name
    model.name = 'resnet50'
    
    print(f"[INFO] Created new {model.name} model.")
    
    return model