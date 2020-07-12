from utils.metric import accuracy

import torch
from tqdm.auto import tqdm
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch.nn as nn

from collections import defaultdict

from agents.exp_replay import Memory
from types import MethodType

import pytorch_lightning as pl
import wandb
import random
import models


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def orthonormalize(vectors, gpu, normalize=True):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    for i in tqdm(range(1, vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors

def project_vec(vec, proj_basis, gpu):
    if proj_basis.shape[1] > 0:  # param x basis_size
        dots = torch.matmul(vec, proj_basis)  # basis_size
        out = torch.matmul(proj_basis, dots)
        return out
    else:
        return torch.zeros_like(vec)


def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def grad_vector_to_parameters(vec, parameters):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param



def validate(testloader, model, gpu, size):
    model.eval()

    correct = 0
    total = 0
    acc = 0
    acc_cnt = 0
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            if idx < size:
                data, target, task = data
                if gpu:
                    with torch.no_grad():
                        # data = data.cuda().double()
                        data = data.cuda()
                        target = target.cuda()

                outputs = model.forward(data, task)

                acc += accuracy(outputs, target)
                acc_cnt += 1

            else:
                break
    return acc / acc_cnt


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


class AgentModel(pl.LightningModule):
    def __init__(self, config, wandb_run, val_loaders):
        super().__init__()
        self.criterion_fn = nn.CrossEntropyLoss()
        self.optimizer = None
        self.config = config
        self.wandb_run = wandb_run

        self.model = self.create_model()

        # wandb.watch(self.model)
        print(f"### The model has {count_parameter(self.model)} parameters ###")

        if self.config.is_split :
            n_params = count_parameter(self.model.linear)
        else :
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        # self.ogd_basis = None
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]))
        self.mem_dataset = None

        if self.config.gpu:
            # self.ogd_basis = self.ogd_basis.cuda()
            self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())

        self.val_loaders = val_loaders

        self.task_count = 0
        self.task_memory = {}
        self.task_mem_cache = {}

        self.task_grad_memory = {}
        self.task_grad_mem_cache = {}

    def create_model(self):
        # Seems like pulling the model and adding heads if necessary

        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        # in_channel = 1 if self.config.dataset == "MNIST" else 3
        if cfg.model_type not in ["mlp", "lenet"] :
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        else :
            model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim)
                                                                               # in_channel=in_channel)

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output} -> OK ! wow finally understood ! very clever :)
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        # assert list(model.last.keys()) == ["All"] # Routine check to be sure to do the right thing :)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['model_weights'] is not None:
            print('=> Load model weights:', cfg['model_weights'])
            model_state = torch.load(cfg['model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state)
            print('=> Load Done')
        return model

    def _get_new_ogd_basis(self, train_loader, gpu, optimizer, model):
        new_basis = []

        for i, (inputs, targets, tasks) in tqdm(enumerate(train_loader),
                                                desc="_get_new_ogd_basis",
                                                total=len(train_loader.dataset)):
            # if i < memory_size:
            if gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # out = model(inputs)
            out = self.forward(x=inputs, task=(tasks))
            # assert out.shape[0] == 1
            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            # grad_vec = parameters_to_grad_vector(model.parameters())
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
            new_basis.append(grad_vec)
            # else:
            #     print('Memory filled')
            #     break
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    def forward(self, x, task):
        # TODO : Check :-)
        task_key = task[0]
        out = self.model.forward(x)
        if self.config.is_split :
            return out[task_key]
        else :
            return out["All"]

    def training_step(self, batch, batch_nb):
        self.model.train()
        assert self.model.training

        data, target, task = batch
        output = self.forward(data, task)
        loss = self.criterion_fn(output, target)

        self.task_id = int(task[0])
        # self.task_id = task_id

        if batch_nb % self.config.val_check_interval == 0:
            log_dict = dict()
            for task_id_val in range(1, self.task_id + 1):
                val_acc = validate(self.val_loaders[task_id_val - 1],
                                   model=self,
                                   gpu=self.config.gpu,
                                   size=self.config.val_size)
                log_dict[f"val_acc_{task_id_val}"] = val_acc
            wandb.log(log_dict,
                      commit=False)

        wandb.log({f"train_loss_{self.task_id}": loss,
                   "task_id": self.task_id})

        return {'loss': loss}

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self.config.lr,
                                         momentum=0,
                                         weight_decay=0)
        return self.optimizer

    def get_params_dict(self, last, task_key=None):
        if self.config.is_split :
            if last:
                return self.model.last[task_key].parameters()
            else:
                return self.model.linear.parameters()
        else:
            return self.model.parameters()

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None, using_native_amp=None):
        # STEP :
        # TODO : params_dict under conditions, check that the updates are done correctly

        task_key = str(self.task_id)

        cur_param = parameters_to_vector(self.get_params_dict(last=False))
        grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
        if self.config.ogd or self.config.ogd_plus:
            proj_grad_vec = project_vec(grad_vec,
                                        proj_basis=self.ogd_basis,
                                        gpu=self.config.gpu)
            new_grad_vec = grad_vec - proj_grad_vec
        else:
            new_grad_vec = grad_vec

        cur_param -= self.config.lr * new_grad_vec
        vector_to_parameters(cur_param, self.get_params_dict(last=False))

        if self.config.is_split :
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True, task_key=task_key))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, task_key=task_key))
            cur_param -= self.config.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True, task_key=task_key))
            # TODO : Add GD for the last layer :)
        # ZERO GRAD
        optimizer.zero_grad()

    def _update_mem(self, data_train_loader, val_loader=None):
        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1

        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.config.memory_size // self.task_count
        wandb.log({"num_sample_per_task": num_sample_per_task}, commit=False)
        num_sample_per_task = min(len(data_train_loader.dataset), num_sample_per_task)

        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)

        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(data_train_loader.dataset[ind])

        # (d) Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=1)
            assert len(mem_loader) == 1, 'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.config.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data': mem_input, 'target': mem_target, 'task': mem_task}

        ####################################### Grads MEM ###########################

        for storage in self.task_grad_memory.values():
            storage.reduce(num_sample_per_task)

        # (e) Get the new non-orthonormal gradients basis
        if self.config.ogd:
            ogd_train_loader = torch.utils.data.DataLoader(self.task_memory[self.task_count],
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=1)
        elif self.config.ogd_plus:
            all_task_memory = []
            for task_id, mem in self.task_memory.items():
                all_task_memory.extend(mem)
            random.shuffle(all_task_memory)
            ogd_memory_list = all_task_memory[:num_sample_per_task]
            ogd_memory = Memory()
            for obs in ogd_memory_list:
                ogd_memory.append(obs)
            ogd_train_loader = torch.utils.data.DataLoader(ogd_memory,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=1)

        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader,
                                              self.config.gpu,
                                              self.optimizer,
                                              self.model)

        # (f) Ortonormalise the whole memorized basis
        if self.config.is_split:
            n_params = count_parameter(self.model.linear)
        else:
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)

        if self.config.gpu:
            self.ogd_basis = self.ogd_basis.cuda()

        for t, mem in self.task_grad_memory.items():
            # Concatenate all data in each task
            task_ogd_basis_tensor = mem.get_tensor()
            if self.config.gpu:
                task_ogd_basis_tensor = task_ogd_basis_tensor.cuda()

            self.ogd_basis = torch.cat([self.ogd_basis, task_ogd_basis_tensor], axis=1)

        self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)

        self.ogd_basis = orthonormalize(self.ogd_basis, gpu=self.config.gpu, normalize=True)

        # (g) Store in the new basis
        ptr = 0
        for t, mem in self.task_memory.items():
            task_mem_size = len(mem)

            idxs_list = [i + ptr for i in range(task_mem_size)]
            if self.config.gpu:
                self.ogd_basis_ids[t] = torch.LongTensor(idxs_list).cuda()
            else:
                self.ogd_basis_ids[t] = torch.LongTensor(idxs_list)

            self.task_grad_memory[t] = Memory()  # Initialize the memory slot
            for ind in range(task_mem_size):  # save it to the memory
                self.task_grad_memory[t].append(self.ogd_basis[:, ptr])
                ptr += 1
        print(f"Used memory {ptr} / {self.config.memory_size}")
        # assert ptr == self.config.memory_size

    def update_ogd_basis(self, task_id, data_train_loader):
        if self.config.ogd or self.config.ogd_plus:
            self._update_mem(data_train_loader)
