from ogd.tools import *
import pytorch_lightning as pl


class AgentModel(pl.LightningModule):
    def __init__(self, config, wandb_run, val_loaders):
        super().__init__()
        self.criterion_fn = nn.CrossEntropyLoss()

        self.config = config
        self.wandb_run = wandb_run

        self.model = self.create_model()
        print(f"### The model has {count_parameter(self.model)} parameters ###")

        # # TODO : remove from init : added only for the NTK gen part ?
        # self.optimizer = self.optimizer = torch.optim.SGD(params=self.model.parameters(),
        #                                                   lr=self.config.lr,
        #                                                   momentum=0,
        #                                                   weight_decay=0)

        if self.config.is_split_cub :
            n_params = get_n_trainable(self.model)
        elif self.config.is_split :
            n_params = count_parameter(self.model.linear)
        else :
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        # self.ogd_basis = None
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]))

        if self.config.gpu:
            # self.ogd_basis = self.ogd_basis.cuda()
            self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())

        self.val_loaders = val_loaders

        # Store initial Neural Tangents

        self.task_count = 0
        self.task_memory = {}
        self.task_mem_cache = {}

        self.task_grad_memory = {}
        self.task_grad_mem_cache = {}

        self.mem_loaders = list()

    def create_model(self):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        # in_channel = 1 i f self.config.dataset == "MNIST" else 3

        # if cfg.model_type not in ["alexnet"] :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # if cfg.model_type not in ["mlp", "lenet"] :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']]()
        # else :
        #     model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim,
        #                                                                            dropout=self.config.dropout)
                                                                               # in_channel=in_channel)
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](hidden_dim=self.config.hidden_dim)

        n_feat = model.last.in_features
        model.last = nn.ModuleDict()
        for task,out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat,out_dim)

        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)

        if self.config.gpu :
            device = torch.device("cuda")
            model.to(device)
        return model

    def _get_new_ogd_basis(self, train_loader, last=False):
        return self._get_neural_tangents(train_loader,
                                         gpu=self.config.gpu,
                                         optimizer=self.optimizer,
                                         model=self.model, last=last)

    def _get_neural_tangents(self, train_loader, gpu, optimizer, model, last):
        new_basis = []

        for i, (inputs, targets, tasks) in tqdm(enumerate(train_loader),
                                                desc="get neural tangents",
                                                total=len(train_loader.dataset)):
            # if gpu:
            inputs = self.to_device(inputs)
            targets = self.to_device(targets)

            out = self.forward(x=inputs, task=(tasks))
            label = targets.item()
            pred = out[0, label]

            optimizer.zero_grad()
            pred.backward()

            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=last))
            new_basis.append(grad_vec)
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    def forward(self, x, task):
        task_key = task[0]
        out = self.model.forward(x)
        if self.config.is_split :
            return out[task_key]
        else :
            return out["All"]

    def predict(self, x, task):
        x = torch.tensor(x)
        x = self.to_device(x)
        out = self.forward(x, task)
        _, pred = out.topk(1, 1, True, True)
        return pred

    def training_step(self, batch, batch_nb):
        self.model.train()
        assert self.model.training

        data, target, task = batch
        output = self.forward(data, task)
        loss = self.criterion_fn(output, target)

        self.task_id = int(task[0])
        if batch_nb % self.config.val_check_interval == 0 and not self.config.no_val:
            log_dict = dict()

            # print(f"keys {self.val_loaders.keys()}, {self.mem_loaders.keys()}")

            for task_id_val in range(1, self.task_id + 1):
                val_acc = validate(self.val_loaders[task_id_val - 1],
                                   model=self,
                                   gpu=self.config.gpu,
                                   size=self.config.val_size)
                log_dict[f"val_acc_{task_id_val}"] = val_acc

            if self.config.ogd or self.config.ogd_plus :
                for task_id_val in range(1, self.task_id):
                    # TODO : Add mem val acc
                    val_acc = validate(self.mem_loaders[task_id_val - 1],
                                       model=self,
                                       gpu=self.config.gpu,
                                       size=self.config.val_size)
                    log_dict[f"mem_val_acc_{task_id_val}"] = val_acc


            # print(log_dict)
            log_dict["task_id"] = self.task_id
            # wandb.log(log_dict,
            #           commit=False)
            wandb.log(log_dict)

        # wandb.log({f"train_loss_{self.task_id}": loss,
        #            "task_id": self.task_id})

        return {'loss': loss}

    def configure_optimizers(self):
        # Unused :)
        n_trainable = get_n_trainable(self.model)
        print(f"The model has {n_trainable} trainable parameters")
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.config.lr,
                                             momentum=0,
                                             weight_decay=0)
        return self.optimizer

    def get_params_dict(self, last, task_key=None):
        if self.config.is_split_cub :
            if last :
                return self.model.last[task_key].parameters()
            else :
                return self.model.linear.parameters()
        elif self.config.is_split :
            if last:
                return self.model.last[task_key].parameters()
            else:
                return self.model.linear.parameters()
        else:
            return self.model.parameters()

    def to_device(self, tensor):
        if self.config.gpu :
            return tensor.cuda()
        else :
            return tensor

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None, using_native_amp=None):
        task_key = str(self.task_id)

        cur_param = parameters_to_vector(self.get_params_dict(last=False))
        grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
        if self.config.ogd or self.config.ogd_plus:
            proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis, gpu=self.config.gpu)
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
        optimizer.zero_grad()

    def _update_mem(self, data_train_loader, val_loader=None):
        # 2.Randomly decide the images to stay in the memory
        self.task_count += 1

        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.config.memory_size

        # (c) Randomly choose some samples from new task and save them to the memory
        self.task_memory[self.task_count] = Memory()  # Initialize the memory slot
        randind = torch.randperm(len(data_train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory
            self.task_memory[self.task_count].append(data_train_loader.dataset[ind])

        ####################################### Grads MEM ###########################

        # (e) Get the new non-orthonormal gradients basis
        if self.config.ogd:
            ogd_train_loader = torch.utils.data.DataLoader(self.task_memory[self.task_count], batch_size=1,
                                                           shuffle=False, num_workers=1)
        elif self.config.ogd_plus:
            all_task_memory = []
            for task_id, mem in self.task_memory.items():
                all_task_memory.extend(mem)
            # random.shuffle(all_task_memory)
            # ogd_memory_list = all_task_memory[:num_sample_per_task]
            ogd_memory_list = all_task_memory
            ogd_memory = Memory()
            for obs in ogd_memory_list:
                ogd_memory.append(obs)
            ogd_train_loader = torch.utils.data.DataLoader(ogd_memory, batch_size=1, shuffle=False, num_workers=1)
        # Non orthonormalised basis
        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader)
        print(f"new_basis_tensor shape {new_basis_tensor.shape}")

        # (f) Ortonormalise the whole memorized basis
        if self.config.is_split:
            n_params = count_parameter(self.model.linear)
        else:
            n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0)
        self.ogd_basis = self.to_device(self.ogd_basis)

        if self.config.ogd :
            for t, mem in self.task_grad_memory.items():
                # Concatenate all data in each task
                task_ogd_basis_tensor = mem.get_tensor()
                task_ogd_basis_tensor = self.to_device(task_ogd_basis_tensor)
                self.ogd_basis = torch.cat([self.ogd_basis, task_ogd_basis_tensor], axis=1)
            self.ogd_basis = torch.cat([self.ogd_basis, new_basis_tensor], axis=1)
        elif self.config.ogd_plus :
            if self.config.pca :
                for t, mem in self.task_grad_memory.items():
                    # Concatenate all data in each task
                    task_ogd_basis_tensor = mem.get_tensor()
                    task_ogd_basis_tensor = self.to_device(task_ogd_basis_tensor)

                    # task_ogd_basis_tensor.shape
                    # Out[3]: torch.Size([330762, 50])
                    start_idx = t * num_sample_per_task
                    end_idx = (t + 1) * num_sample_per_task
                    before_pca_tensor = torch.cat([task_ogd_basis_tensor, new_basis_tensor[:, start_idx:end_idx]], axis=1)
                    u, s, v = torch.svd(before_pca_tensor)

                    # u.shape
                    # Out[8]: torch.Size([330762, 150]) -> col size should be 2 * num_sample_per_task

                    after_pca_tensor = u[:, :num_sample_per_task]

                    # after_pca_tensor.shape
                    # Out[13]: torch.Size([330762, 50])

                    self.ogd_basis = torch.cat([self.ogd_basis, after_pca_tensor], axis=1)
            #   self.ogd_basis.shape should be T * num_sample_per_task

            else :
                self.ogd_basis = new_basis_tensor

        # TODO : Check if start_idx is correct :)
        start_idx = (self.task_count - 1) * num_sample_per_task
        # print(f"the start idx of orthonormalisation if {start_idx}")
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

        if self.config.ogd or self.config.ogd_plus :
            loader = torch.utils.data.DataLoader(self.task_memory[self.task_count],
                                                                            batch_size=self.config.batch_size,
                                                                            shuffle=True,
                                                                            num_workers=2)
            self.mem_loaders.append(loader)

    def update_ogd_basis(self, task_id, data_train_loader):
        if self.config.gpu :
            device = torch.device("cuda")
            self.model.to(device)
        print(f"\nself.model.device update_ogd_basis {next(self.model.parameters()).device}")
        if self.config.ogd or self.config.ogd_plus:
            self._update_mem(data_train_loader)