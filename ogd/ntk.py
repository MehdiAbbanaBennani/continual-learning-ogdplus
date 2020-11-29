import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from dataloaders.wrapper import CacheClassLabel
from torch.utils.data import TensorDataset, DataLoader
from dataloaders.wrapper import AppendName, Subclass

from tqdm import tqdm


def get_subset(train_dataset, n_samples) :
    if isinstance(train_dataset.dataset, Subclass):
        n_labels = len(train_dataset.dataset.class_list)
    else :
        n_labels = len(list(set(train_dataset.dataset.labels.tolist())))


    subset_size = n_samples // n_labels

    cnt = defaultdict(lambda: subset_size)
    new_data = list()
    new_targets = list()
    for i in range(len(train_dataset)) :
        x, label, task_id = train_dataset[i]
        if cnt[label] > 0:
            new_data.append(x)
            new_targets.append(label)
            cnt[label] -= 1
            n_samples -= 1
        if n_samples == 0 :
            break

    indices = np.argsort(new_targets)
    new_data = [new_data[idx].numpy() for idx in indices]
    new_targets = [new_targets[idx] for idx in indices]
    dataset = TensorDataset(torch.Tensor(new_data),
                            torch.Tensor(new_targets).long())
    dataset = AppendName(dataset, name=task_id)
    return dataset, new_data, new_targets


def _build_ntk_loader(data_loader, n_samples):
    subset, new_data, new_targets = get_subset(data_loader.dataset, n_samples)
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
    return loader, subset, new_data, new_targets


class NTK :
    def __init__(self, loaders, kernel_size):
        self.kernel_size = kernel_size

        self.loaders = list()
        self.subsets = list()
        self.inputs_list = list()
        self.labels_list = list()

        for data_loader in tqdm(loaders) :
            loader, subset, new_data, new_targets = _build_ntk_loader(data_loader, kernel_size)

            new_targets = np.array(new_targets)

            self.loaders.append(loader)
            self.subsets.append(subset)
            self.inputs_list.append(new_data)
            self.labels_list.append(new_targets)

    def predict(self, neural_tangents):
        X = neural_tangents.cpu().numpy().T
        return np.dot(X, X.T)
        # return cosine_similarity(X, X)
