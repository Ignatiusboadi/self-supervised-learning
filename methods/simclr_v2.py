import torch.nn as nn
import torchvision.models as models


class SimCLRModelv2(nn.Module):
    resnets = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
               "resnet101": models.resnet101, "resnet152": models.resnet152}
    def __init__(self, model_name, projection_dim=128):
        super().__init__()
        base_model = SimCLRModelv2.resnets[model_name](weights=None)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.encoder = base_model
        self.fc1 = nn.Linear(num_ftrs, projection_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
