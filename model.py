import torch.nn as nn

class KanaModel(nn.Module):
    def __init__(self, in_channels, d_model, out_base, out_dakuten):
        super().__init__()

        out_shared = 512

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_model),
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

        self.script = nn.Linear(out_shared, 2) # 0 - hiragana, 1 - katakana
        self.base = nn.Linear(out_shared, out_base)
        self.fc_dakuten = nn.Linear(out_shared, out_dakuten)


    def forward(self, x):
        shared = self.features(x)
        out_script = self.script(shared)
        out_base = self.base(shared)
        out_dakuten = self.fc_dakuten(shared)

        return out_script, out_base, out_dakuten