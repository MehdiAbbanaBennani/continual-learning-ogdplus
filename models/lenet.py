import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=500):
        super(LeNet, self).__init__()
        feat_map_sz = img_sz//4
        self.n_feat = 50 * feat_map_sz * feat_map_sz
        self.hidden_dim = hidden_dim

        self.linear = nn.Sequential(
            nn.Conv2d(in_channel, 20, 5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(self.n_feat, hidden_dim),
            # TODO : Check why bug when using batchnorm with OGD+
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.last = nn.Linear(hidden_dim, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        # x = self.conv(x)
        # x = self.linear(x.view(-1, self.n_feat))
        return self.linear(x)

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def LeNetC(hidden_dim, out_dim=10):  # LeNet with color input
    return LeNet(out_dim=out_dim, in_channel=3, img_sz=32, hidden_dim=hidden_dim)


if __name__ == '__main__':
    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())

    model = LeNetC(hidden_dim=100)
    n_params = count_parameter(model)
    print(f"LeNetC has {n_params} parameters")
