import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128, hidden_dim=None, num_layers=2, ):
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError("num_layers must be 2 or 3.")

        if input_dim <= 0 or projection_dim <= 0:
            raise ValueError("input_dim and projection_dim must be positive.")

        if hidden_dim is None:
            hidden_dim = input_dim

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        layers = []

        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(), ])

        layers.append(nn.Linear(hidden_dim, projection_dim, bias=True))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        return self.net(x)
