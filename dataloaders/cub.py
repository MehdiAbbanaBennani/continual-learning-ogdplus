import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torch.nn as nn

import gdown


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    url = 'https://drive.google.com/uc?export=download&id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # download_url(self.url, self.root, self.filename, self.tgz_md5)
        output = os.path.join(self.root, self.filename)
        gdown.download(self.url, output, quiet=False)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


import torchvision
from torchvision import transforms
from dataloaders.wrapper import CacheClassLabel
import torchvision.transforms.functional as TF
import torch
from torch.nn import Dropout


def CUB(dataroot, train_aug=False):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        # TODO : Check sizes
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = val_transform

    train_dataset = Cub2011(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = Cub2011(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset

from models.alexnet import AlexNetCL

if __name__ == '__main__':
    import os

    root = "/tmp/datasets"
    dataset = Cub2011(root=root)

    from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
    from tqdm import tqdm

    dataroot = "/tmp/datasets"
    first_split_size = 5
    other_split_size = 5
    rand_split = False
    no_class_remap = False

    # from dataloaders.base import MNIST
    # Dataset = MNIST
    Dataset = CUB
    train_dataset, val_dataset = Dataset(dataroot)
    train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                           first_split_sz=first_split_size,
                                                                           other_split_sz=other_split_size,
                                                                           rand_split=rand_split,
                                                                           remap_class=no_class_remap)
    # __get__ returns img, target, self.name through AppendName

    n_tasks = len(task_output_space.items())
    task_names = sorted(list(task_output_space.keys()), key=int)

    batch_size = 32
    workers = 0

    for i in tqdm(range(len(task_names)), "task"):
        task_name = task_names[i]
        print('======================', task_name, '=======================')
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=workers)

    # AlexNet
    import torch

    ref_dataset = Cub2011(root=dataroot)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.eval()

    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())

    # https://medium.com/@YeseulLee0311/pytorch-transfer-learning-alexnet-how-to-freeze-some-layers-26850fc4ac7e

    print(f"the model has {count_parameter(model)} parameters")

    len(model.classifier)
    print(f"n_features {model.classifier[6].in_features}")


    model = AlexNetCL()
    print(f"LeNetC has {count_parameter(model.linear)} parameters")
    outputs_list = list()
    with torch.no_grad():
        for x, y, task_id in val_loader :
            output = model(x)
            outputs_list.append(output)


    # from models.lenet import LeNetC


    # n_feat = model.last.in_features
    # model.last = nn.ModuleDict()
    # out_dim = first_split_size
    # for task in task_names :
    #     model.last[task] = nn.Linear(n_feat, out_dim)
    #
    # # x_batch, y_batch = next(iter(val_loader))



