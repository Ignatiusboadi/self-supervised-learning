import torch.nn as nn
from models.simclr import SimCLR


class SimCLRClassifier(nn.Module):
    def __init__(self, num_classes=10, **config):
        super().__init__()
        self.backbone = SimCLR(**config)
        self.encoder = self.backbone.encoder
        self.fc = nn.Linear(self.encoder.feature_dim, num_classes, )

    def forward(self, x):
        features = self.encoder(x)
        return self.fc(features)
