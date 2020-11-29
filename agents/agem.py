from ogd.agem import *
from ogd.ntk import NTK

from types import MethodType
from pytorch_lightning import Trainer
import numpy as np


class AGEM:
    def __init__(self, agent_config, wandb_run, val_loaders):
        self.config = agent_config
        self.agent_model = AGEMAgent(config=self.config,
                                     wandb_run=wandb_run,
                                     val_loaders=val_loaders)
        self.task_id = 1

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
        trainer.fit(self.agent_model, train_loader)
        self.agent_model.update_agent_memory(task_id=self.task_id,
                                             data_train_loader=train_loader)
