import torch.nn as nn
import torchvision.models as models

class SimCLRModel(nn.Module):
    resnet_models = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    def __init__(self, resnet_variant="resnet18", projection_dim=128):
        super().__init__()
        self.resnet_variant = resnet_variant
        base_model = SimCLRModel.resnet_models[self.resnet_variant](weights=None)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
