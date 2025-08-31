import torch.nn as nn

class KanaModel(nn.Module):
    def __init__(self, in_channels, d_model, out_base, out_dakuten):
        super().__init__()

        out_shared = 512
        print(d_model)
        self.FiLM = FiLM(in_channels, d_model)

        # separate these to apply FiLM later
        self.conv1 = nn.Conv2d(in_channels, d_model, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(d_model)

        self.features = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(d_model, d_model*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(d_model*2, d_model*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(256 * 6 * 6, out_shared),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.base = nn.Linear(out_shared, out_base)
        self.fc_dakuten = nn.Linear(out_shared, out_dakuten)


    def forward(self, script_onehot, x):
        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)

        shared = self.features(self.FiLM(norm1, script_onehot))

        out_base = self.base(shared)
        out_dakuten = self.fc_dakuten(shared)

        return out_base, out_dakuten

class FiLM(nn.Module):
    def __init__(self, in_channels, d_model):
        super().__init__()

        self.FiLM = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, feature_map, script):
        film = self.FiLM(script)
        gamma, beta = film.chunk(2, dim=1)  # split along features
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # broadcast over H,W
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature_map * (1 + gamma) + beta