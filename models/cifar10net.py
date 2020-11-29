import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self, out_dim=10, in_channel=1, img_sz=32):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 7744

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.n_feat, 512),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(512, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.conv(x)
        x = self.linear(x.view(-1, self.n_feat))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def LeNetC(out_dim=10):  # LeNet with color input
    return LeNet(out_dim=out_dim, in_channel=3, img_sz=32)