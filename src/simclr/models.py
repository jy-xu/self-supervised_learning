import torch.nn as nn
from torchvision import models

# SimCLR Framework with modified ResNet-50 and 2-layer MLP
class SimCLR(nn.Module):
    def __init__(self, base_model, out_ftrs):
        super(SimCLR, self).__init__()
        self.base_encoder_zoo = {"resnet18": models.resnet18(pretrained=False),
                                 "resnet50": models.resnet50(pretrained=False)}
        base_encoder = self._get_base_encoder(base_model)
        in_ftrs = base_encoder.fc.in_features
        self.encoder = self._modify_encoder(base_encoder)
        self.encoder.out_ftrs = in_ftrs
        self.projector = self._make_projector(in_ftrs, out_ftrs)
        
    def _get_base_encoder(self, base_model):
        base_encoder = self.base_encoder_zoo[base_model]
        return base_encoder
    
    def _modify_encoder(self, encoder):
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.conv1 = conv1
        encoder.maxpool = nn.Identity()
        encoder.fc = nn.Identity()
        return encoder
    
    def _make_projector(self, in_ftrs, out_ftrs):
        projector = nn.Sequential()
        projector.add_module('hidden', nn.Linear(in_ftrs, in_ftrs))
        projector.add_module('relu', nn.ReLU(inplace=True))
        projector.add_module('out', nn.Linear(in_ftrs, out_ftrs))
        return projector
    
    def forward(self, xi, xj=None):
        hi = self.encoder(xi)
        zi = self.projector(hi)
        
        if xj is not None:
            hj = self.encoder(xj)
            zj = self.projector(hj)
            return hi, hj, zi, zj
        else:
            return hi, zi

# Modified ResNet for CIFAR10
class ModifiedResNet(nn.Module):
    def __init__(self, base_model, out_ftrs):
        super(ModifiedResNet, self).__init__()
        self.base_encoder_zoo = {"resnet18": models.resnet18(pretrained=False),
                                 "resnet50": models.resnet50(pretrained=False)}
        base_encoder = self._get_base_encoder(base_model)
        self.encoder = self._modify_encoder(base_encoder, out_ftrs)
        
    def _get_base_encoder(self, base_model):
        base_encoder = self.base_encoder_zoo[base_model]
        return base_encoder
    
    def _modify_encoder(self, encoder, out_ftrs):
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.conv1 = conv1
        encoder.maxpool = nn.Identity()
        encoder.fc.out_features = out_ftrs
        return encoder
    
    def forward(self, xi):
        yi = self.encoder(xi)
        return yi

# Simple classifier with 1 linear fc layer for Linear Evaluation
class SimpleNet(nn.Module):
    def __init__(self, input_dim=2048, num_classes=10):
        super(SimpleNet, self).__init__()

        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

# Finetune classifier with SimCLR base encoder with 1 linear fc layer attached
class FinetuneNet(nn.Module):
    def __init__(self, simclr, num_classes=10):
        super(FinetuneNet, self).__init__()
        self.encoder = simclr
        in_ftrs = simclr.encoder.out_ftrs
        self.fc = nn.Linear(in_ftrs, num_classes)

    def forward(self, x):
        out, _ = self.encoder(x)
        out = self.fc(out)
        return out