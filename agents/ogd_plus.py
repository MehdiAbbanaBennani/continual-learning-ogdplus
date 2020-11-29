# from ogd.ogd_core import *
from ogd.ogd_core_new import *
from ogd.ntk import NTK

from types import MethodType
from pytorch_lightning import Trainer
import numpy as np
from utils.metric import accuracy, AverageMeter, Timer

import numpy as np


def compute_gen_bound(kernels_list, labels_list) :
    total_bound = 0
    for kernel, labels in zip(kernels_list, labels_list):
        n_tau = labels.shape[0]
        kernel_inv = np.linalg.inv(kernel)
        cur_bound = np.sqrt(np.dot(np.dot(labels, kernel_inv), labels) * np.trace(kernel) / n_tau**2)
        # cur_bound = np.sqrt(np.dot(np.dot(labels, kernel_inv), labels) / n_tau)
        total_bound += cur_bound
    return total_bound


class OGD:
    def __init__(self, agent_config, wandb_run, val_loaders):
        self.config = agent_config
        self.agent_model = AgentModel(config=self.config,
                                      wandb_run=wandb_run,
                                      val_loaders=val_loaders)
        self.task_id = 1
        # if self.config.is_split :
        #     self.ntk = NTK(loaders=val_loaders, kernel_size=100)
        #
        #     self.kernels_list = list()
        #     self.y_tilde_list = list()
        #     self.y_list = list()
        #     self.y_pred_list = list()
        #     self.x_list = list()
        #     self.gen_bound_list = list()
        #     self.gen_exp_list = list()
    #
    # def gen_step_before(self):
    #     # Append the new kernel
    #     loader = self.ntk.loaders[self.task_id - 1]
    #     neural_tangents = self.agent_model._get_new_ogd_basis(loader, last=False)
    #
    #     if self.agent_model.config.ogd or self.agent_model.config.ogd_plus:
    #         ogd_basis = self.agent_model.ogd_basis
    #     else:
    #         ogd_basis = torch.zeros(neural_tangents.shape[0], 1)
    #         ogd_basis = self.agent_model.to_device(ogd_basis)
    #
    #     if self.task_id == 1 :
    #         kernel = self.ntk.predict(neural_tangents)
    #     else :
    #         projected_neural_tangents = project_vec(vec=neural_tangents.T,
    #                                                 proj_basis=ogd_basis,
    #                                                 gpu=self.agent_model.config.gpu)
    #         kernel = self.ntk.predict(projected_neural_tangents)
    #     self.kernels_list.append(kernel)
    #
    #     # Append the new y_tilde
    #     if self.task_id == 0:
    #         self.y_tilde_list.append(self.ntk.labels_list[0])
    #     else:
    #         cur_inputs = torch.tensor(self.ntk.inputs_list[self.task_id - 1])
    #         cur_inputs = self.agent_model.to_device(cur_inputs)
    #         cur_labels = self.ntk.labels_list[self.task_id - 1]
    #
    #         pred_labels = self.agent_model.predict(cur_inputs, task=[str(self.task_id)])
    #         pred_labels = pred_labels.clone().detach().cpu().numpy().flatten()
    #         y_tilde = cur_labels - pred_labels
    #
    #         self.y_tilde_list.append(y_tilde)
    #         self.y_pred_list.append(pred_labels)
    #         self.y_list.append(cur_labels)
    #         self.x_list.append(cur_inputs)
    #
    #         self.agent_model.optimizer.zero_grad()

        # # Compute the generalisation bound
        # gen_bound = compute_gen_bound(self.kernels_list,
        #                               self.y_tilde_list)
        # self.gen_bound_list.append(gen_bound)
        # wandb.log(dict(gen_bound=gen_bound), commit=False)

    def gen_step_after(self):
        # val_acc = validate(self.agent_model.val_loaders[0],
        #                    model=self.agent_model,
        #                    gpu=self.config.gpu,
        #                    size=self.config.val_size)
        # val_loss = 1 - val_acc / 100
        #
        # self.gen_exp_list.append(val_loss)
        #
        # bound_dict = dict(task_id=self.task_id,
        #                   gen_val_loss=val_loss)
        bound_dict = dict(dummy=0)
        wandb.log(bound_dict, commit=True)
        return

    def learn_batch(self, train_loader, val_loader=None):
        print('======================', self.task_id, '=======================')
        if self.config.gpu:
            gpus = 1
        else:
            gpus = 0

        trainer = Trainer(gpus=gpus,
                          num_nodes=1,
                          limit_train_batches=float(self.config.train_percent_check),
                          num_sanity_val_steps=0,
                          max_epochs=self.config.nepoch)
        # if self.config.is_split :
        #     self.gen_step_before()

        trainer.fit(self.agent_model, train_loader)
        self.agent_model.update_ogd_basis(task_id=self.task_id,
                                          data_train_loader=train_loader)

        if self.config.is_split :
            self.gen_step_after()

        self.task_id += 1

    def validation(self, dataloader):
        acc = validate(dataloader,
                           model=self.agent_model,
                           gpu=self.agent_model.config.gpu,
                           size=None)
        return acc
