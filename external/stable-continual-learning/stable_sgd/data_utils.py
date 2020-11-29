import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TorchVisionFunc
from torchvision import transforms

from stable_sgd.cub import Cub2011

import os
from random import shuffle

SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR')
DATA_ROOT: str = os.path.join(SLURM_TMPDIR, "datasets")
# DATA_ROOT = "/tmp/datasets"


def get_class_map(num_classes, split_size, rand_split):
    split_boundaries = [0, split_size]
    while split_boundaries[-1] < num_classes:
        split_boundaries.append(split_boundaries[-1] + split_size)
    assert split_boundaries[-1] == num_classes, 'Invalid split size'
    
    # Assign classes to each splits
    # Create the dict: {split_name1:[2,6,7], split_name2:[0,3,9], ...}
    if not rand_split:
        class_map = {i: i for i in range(num_classes)}
    else:
        randseq = torch.randperm(num_classes)
        class_lists = {str(i): randseq[list(range(split_boundaries[i - 1], split_boundaries[i]))].tolist() for i in
                       range(1, len(split_boundaries))}
        print(class_lists)
        
        new_classes = list()
        for i in range(1, len(split_boundaries)):
            new_classes.extend(class_lists[str(i)])
        class_map = {i: new_classes[i] for i in range(num_classes)}
    
    return class_map


def get_permuted_mnist(task_id, batch_size):
    """
    Get the dataset loaders (train and test) for a `single` task of permuted MNIST.
    This function will be called several times for each task.
    
    :param task_id: id of the task [starts from 1]
    :param batch_size:
    :return: a tuple: (train loader, test loader)
    """
    
    # convention, the first task will be the original MNIST images, and hence no permutation
    if task_id == 1:
        idx_permute = np.array(range(784))
    else:
        rand_ind = list(range(784))
        shuffle(rand_ind)
        rand_ind = np.array(rand_ind)
        idx_permute = torch.from_numpy(rand_ind)
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute]),
                                                 ])
    mnist_train = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transforms), batch_size=256,
        shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader


def get_permuted_mnist_tasks(num_tasks, batch_size):
    """
    Returns the datasets for sequential tasks of permuted MNIST
    
    :param num_tasks: number of tasks.
    :param batch_size: batch-size for loaders.
    :return: a dictionary where each key is a dictionary itself with train, and test loaders.
    """
    datasets = {}
    for task_id in range(1, num_tasks + 1):
        train_loader, test_loader = get_permuted_mnist(task_id, batch_size)
        datasets[task_id] = {'train': train_loader, 'test': test_loader}
    return datasets


class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    
    def __init__(self, angle):
        self.angle = angle
    
    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, batch_size):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    # FIXME : changed 10 to 5
    per_task_rotation = 5
    # per_task_rotation = 10
    rotation_degree = (task_id - 1) * per_task_rotation
    rotation_degree -= (np.random.random() * per_task_rotation)
    
    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
    ])
    
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transforms), batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transforms), batch_size=256,
        shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader


def get_rotated_mnist_tasks(num_tasks, batch_size):
    """
    Returns data loaders for all tasks of rotation MNIST dataset.
    :param num_tasks: number of tasks in the benchmark.
    :param batch_size:
    :return:
    """
    datasets = {}
    for task_id in range(1, num_tasks + 1):
        train_loader, test_loader = get_rotated_mnist(task_id, batch_size)
        datasets[task_id] = {'train': train_loader, 'test': test_loader}
    return datasets


def get_split_cifar100(task_id, batch_size, cifar_train, cifar_test):
    """
    Returns a single task of split CIFAR-100 dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    
    start_class = (task_id - 1) * 5
    end_class = task_id * 5
    
    targets_train = torch.tensor(cifar_train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    
    targets_test = torch.tensor(cifar_test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx == 1)[0]), batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx == 1)[0]), batch_size=batch_size)
    
    return train_loader, test_loader


def get_split_cifar100_tasks(num_tasks, batch_size):
    """
    Returns data loaders for all tasks of split CIFAR-100
    :param num_tasks:
    :param batch_size:
    :return:
    """
    datasets = {}
    
    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    cifar_train = torchvision.datasets.CIFAR100(DATA_ROOT, train=True, download=True, transform=cifar_transforms)
    cifar_test = torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=cifar_transforms)
    
    for task_id in range(1, num_tasks + 1):
        train_loader, test_loader = get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
        datasets[task_id] = {'train': train_loader, 'test': test_loader}
    return datasets


def get_split_cub200(task_id, batch_size, cub_train, cub_test, task_classes = 20):
    """
    Returns a single task of split CUB-200 dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    start_class = (task_id - 1) * task_classes
    end_class = task_id * task_classes
    
    targets_train = cub_train.targets
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    
    targets_test = cub_test.targets
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(cub_train, np.where(target_train_idx == 1)[0]), batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(cub_test, np.where(target_test_idx == 1)[0]), batch_size=batch_size)
    
    return train_loader, test_loader


def get_split_cub200_tasks(num_tasks, batch_size):
    """
    Returns data loaders for all tasks of split CIFAR-100
    :param num_tasks:
    :param batch_size:
    :return:
    """
    datasets = {}
    
    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    
    cub_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cub_train = Cub2011(root=DATA_ROOT,
                        train=True,
                        download=True,
                        transform=cub_transforms
                        )
    cub_test = Cub2011(
        DATA_ROOT,
        train=False,
        transform=cub_transforms
    )
    
    # assert num_tasks == 10
    num_classes = 200
    split_size = num_classes // num_tasks
    
    # Shuffle labels :
    class_map = get_class_map(num_classes=num_classes, split_size=split_size, rand_split=False)
    cub_train.targets = list(map(lambda x: class_map[x], cub_train.targets))
    cub_train.targets = torch.tensor(cub_train.targets)

    cub_test.targets = list(map(lambda x: class_map[x], cub_test.targets))
    cub_test.targets = torch.tensor(cub_test.targets)
        # cub_train.targets.apply_()
    # cub_test.targets = cub_test.targets.apply_(lambda x: class_map[x])
    
    for task_id in range(1, num_tasks + 1):
        train_loader, test_loader = get_split_cub200(task_id, batch_size, cub_train, cub_test)
        datasets[task_id] = {'train': train_loader, 'test': test_loader}
    return datasets