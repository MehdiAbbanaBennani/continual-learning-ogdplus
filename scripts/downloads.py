import gdown

url = 'https://drive.google.com/uc?export=download&id=1WxFZQyt3v7QRHwxFbdb1KO02XWLT0R9z'
output = "Permuted_Omniglot_task50.pt"
gdown.download(url, output, quiet=False)



from dataloaders.cub import Cub2011

root = "/scratch/thang/iclr-2021/datasets"
dataset = Cub2011(root=root)


import gdown
url = 'https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
output = "CUB_200_2011.tgz"
gdown.download(url, output, quiet=False)


import torch
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)