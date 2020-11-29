from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from torch.nn import Dropout


class AlexNetCL(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=10):
        super(AlexNetCL, self).__init__()

        self.alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.alexnet.eval()
        
        # Freeze all layers
        for param in self.alexnet.parameters() :
            param.requires_grad = False

        # https://medium.com/@YeseulLee0311/pytorch-transfer-learning-alexnet-how-to-freeze-some-layers-26850fc4ac7e
        # model.classifier
        # Out[10]:
        # Sequential(
        # (0): Dropout(p=0.5, inplace=False)
        # (1): Linear(in_features=9216, out_features=4096, bias=True)
        # (2): ReLU(inplace=True)
        # (3): Dropout(p=0.5, inplace=False)
        # (4): Linear(in_features=4096, out_features=4096, bias=True)
        # (5): ReLU(inplace=True)
        # (6): Linear(in_features=4096, out_features=1000, bias=True)
        # )

        # # TOOO : Maybe start from the next layer -> less parameters to fit :)
        # self.linear = nn.Sequential(
        #     nn.Linear(in_features=9216, out_features=256, bias=True),
        #     nn.ReLU(inplace=True),
        #     Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(in_features=256, out_features=out_dim, bias=True),
        # )

        # TOOO : Maybe start from the next layer -> less parameters to fit :)
        self.linear = nn.Sequential(
            nn.Linear(in_features=4096, out_features=hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=hidden_dim, out_features=128, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(in_features=256, out_features=out_dim, bias=True),
        )

        self.last = nn.Linear(128, out_dim)  # Subject to be replaced dependent on task

    def features(self, x):
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier[:4](x)
        x = self.linear(x)
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


if __name__ == '__main__':
    model = AlexNetCL()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {n_trainable} trainable parameters")
    #
    # # Download an example image from the pytorch website
    # import urllib
    #
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # try:
    #     urllib.URLopener().retrieve(url, filename)
    # except:
    #     urllib.request.urlretrieve(url, filename)
    #
    # # sample execution (requires torchvision)
    # from PIL import Image
    # from torchvision import transforms
    #
    # input_image = Image.open(filename)
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    #
    # # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')
    #
    # with torch.no_grad():
    #     output = model(input_batch)
    # # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # print(torch.nn.functional.softmax(output[0], dim=0))