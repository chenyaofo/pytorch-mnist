import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifer = nn.Sequential(
            nn.Linear(16 * 4 * 4, 128),
            nn.Linear(128,10),
        )


    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 16 * 4 * 4)
        out = self.classifer(out)
        return out
