from .ogd_core_new import AgentModel
from ogd.tools import *


class AGEMAgent(AgentModel) :
    def __init__(self, config, wandb_run, val_loaders):
        super().__init__(config=config,
                         wandb_run=wandb_run,
                         val_loaders=val_loaders)

        self.agem_mem = list()
        self.agem_mem_loader = None

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience

    def update_agent_memory(self, task_id, data_train_loader):
        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1

        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.config.memory_size

        # (c) Randomly choose some samples from new task and save them to the memory
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            data, target, task = data_train_loader.dataset[ind]
            # data = self.to_device(data)
            # target = self.to_device(torch.tensor(target))
            sample = (data, target, task)
            self.agem_mem.append(sample)

        mem_loader_batch_size = min(self.config.agem_mem_batch_size, len(self.agem_mem))
        self.agem_mem_loader = torch.utils.data.DataLoader(self.agem_mem,
                                                           batch_size=mem_loader_batch_size,
                                                           shuffle=True,
                                                           num_workers=self.config.workers)
        print(f"config.agem_mem_batch_size {self.config.agem_mem_batch_size}")
        print(f"mem_loader_batch_size {mem_loader_batch_size}")
        print(f"self.agem_mem {len(self.agem_mem)}")
        # for i, (mem_input, mem_target, mem_task) in enumerate(self.agem_mem_loader):
        #     if self.gpu:
        #         mem_input = mem_input.cuda()
        #         mem_target = mem_target.cuda()
        #     self.task_mem_cache[i] = (mem_input, mem_target, mem_task)

    def _project_agem_grad(self, batch_grad_vec, mem_grad_vec):
        if torch.dot(batch_grad_vec, mem_grad_vec) >= 0 and not self.config.no_transfer :
            # print(f"\n pos device {mem_grad_vec.device}")
            return batch_grad_vec
        else :
            frac = torch.dot(batch_grad_vec, mem_grad_vec) / torch.dot(mem_grad_vec, mem_grad_vec)
            # print(f"frac {frac}")
            new_grad = batch_grad_vec - frac * mem_grad_vec
            # print(f"\n device {new_grad.device}")
            # print(f"{torch.dot(new_grad, mem_grad_vec)}")
            check = torch.dot(new_grad, mem_grad_vec)
            # assert torch.abs(check) < 1e-5
            return new_grad

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None, using_native_amp=None):
        # self.model.train()
        # assert self.model.training

        task_key = str(self.task_id)

        # Update the last layer with the precomputed gradient
        if self.config.is_split :
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True, task_key=task_key))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, task_key=task_key))
            cur_param -= self.config.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True, task_key=task_key))

        # Get the gradient of the up to last layer
        batch_grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
        cur_param = parameters_to_vector(self.get_params_dict(last=False))

        # Sample a batch from the memory
        if self.agem_mem_loader is not None :
            optimizer.zero_grad()
            data, target, task = next(iter(self.agem_mem_loader))
            data = self.to_device(data)
            target = self.to_device(target)
            output = self.forward(data, task)
            mem_loss = self.criterion_fn(output, target)
            mem_loss.backward()
            mem_grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
            new_grad_vec = self._project_agem_grad(batch_grad_vec=batch_grad_vec,
                                                   mem_grad_vec=mem_grad_vec)
        else :
            new_grad_vec = batch_grad_vec
        cur_param -= self.config.lr * new_grad_vec
        vector_to_parameters(cur_param, self.get_params_dict(last=False))

        optimizer.zero_grad()
        # assert self.model.training
