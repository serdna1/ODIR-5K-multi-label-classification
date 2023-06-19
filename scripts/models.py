import torchvision
import torch
from torch import nn

class NCNV0(nn.Module):
    def __init__(self, backbone, classifier_in_shape, classifier_out_shape):
        super(NCNV0, self).__init__()
        self.backbone = backbone
        self.classifier =  nn.Linear(classifier_in_shape*2, classifier_out_shape)
    
    def forward(self, x_left, x_right):
        x_left = self.backbone(x_left)
        x_right = self.backbone(x_right)
        x_cat = torch.cat((x_left, x_right), dim=1)
        x = self.classifier(x_cat)
        
        return x
    
class NCNV1(nn.Module):
    def __init__(self, backbone, classifier_in_shape, classifier_out_shape):
        super(NCNV1, self).__init__()
        self.backbone = backbone
        self.classifier =  nn.Sequential(nn.Dropout(p=0.2),
                                         nn.Linear(classifier_in_shape*2, classifier_out_shape))
    
    def forward(self, x_left, x_right):
        x_left = self.backbone(x_left)
        x_right = self.backbone(x_right)
        x_cat = torch.cat((x_left, x_right), dim=1)
        x = self.classifier(x_cat)
        
        return x
    
class NCNV2(nn.Module):
    def __init__(self, backbone, classifier_in_shape, classifier_out_shape):
        super(NCNV2, self).__init__()
        self.backbone = backbone
        self.classifier =  nn.Sequential(nn.Linear(classifier_in_shape*2, 128),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(128, 64),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.5),
                                         nn.Linear(64, classifier_out_shape))
    
    def forward(self, x_left, x_right):
        x_left = self.backbone(x_left)
        x_right = self.backbone(x_right)
        x_cat = torch.cat((x_left, x_right), dim=1)
        x = self.classifier(x_cat)
        
        return x

def create_resnet50_dual(version=0):
    # Import a resnet50 from pytorch
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    backbone = torchvision.models.resnet50(weights=weights)

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Get the input shape of the fc layer
    classifier_in_shape = backbone.fc.in_features

    # Remove the fc layer
    backbone.fc = torch.nn.Identity()

    # Create an NCN model with a resnet50 feature extractor as backbone
    if version == 0:
        model = NCNV0(backbone=backbone,
                      classifier_in_shape=classifier_in_shape,
                      classifier_out_shape=8)
        model.name = 'resnet50_dual'
    elif version == 1:
        model = NCNV1(backbone=backbone,
                      classifier_in_shape=classifier_in_shape,
                      classifier_out_shape=8)
        model.name = 'resnet50_dual_v1'
    elif version == 2:
        model = NCNV2(backbone=backbone,
                      classifier_in_shape=classifier_in_shape,
                      classifier_out_shape=8)
        model.name = 'resnet50_dual_v2'

    print(f"[INFO] Created new {model.name} model.")

    return model
