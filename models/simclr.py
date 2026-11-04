import torch.nn as nn

from models.resnet import SimCLRResNetEncoder
from models.projection_head import ProjectionHead


class SimCLR(nn.Module):
    def __init__(self, resnet_variant="resnet18", width_multiplier=1, use_sk=False,
        small_image_stem=False, in_channels=3, projection_dim=128, projection_layers=2,
        projection_hidden_dim=512, zero_init_residual=False,):
        super().__init__()

        self.encoder = SimCLRResNetEncoder(resnet_variant=resnet_variant, width_multiplier=width_multiplier,
                                           use_sk=use_sk, in_channels=in_channels, small_image_stem=small_image_stem,
                                           zero_init_residual=zero_init_residual)
        self.projector = ProjectionHead(input_dim=self.encoder.feature_dim, projection_dim=projection_dim,
                                        hidden_dim=projection_hidden_dim, num_layers=projection_layers)

        self.feature_dim = self.encoder.feature_dim
        self.projection_dim = projection_dim

    def encode(self, x):
        return self.encoder(x)

    def project(self, h):
        return self.projector(h)

    def forward(self, x, return_features=False):
        h = self.encode(x)
        z = self.project(h)

        if return_features:
            return h, z

        return z

    def get_config(self):
        return {"resnet_variant": self.encoder.resnet_variant, "width_multiplier": self.encoder.width_multiplier,
                "use_sk": self.encoder.use_sk, "small_image_stem": self.encoder.small_image_stem,
                "feature_dim": self.feature_dim, "projection_dim": self.projection_dim,
                "projection_layers": self.projector.num_layers, "projection_hidden_dim": self.projector.hidden_dim}
