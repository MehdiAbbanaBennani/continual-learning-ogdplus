import os
SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR')

import torch
from random import shuffle
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import agents
from utils.utils import dotdict

import wandb

from tap import Tap
import numpy as np
import os
from tqdm import tqdm
from typing import List
import time


def prepare_dataloaders(args):
    # Prepare dataloaders
    Dataset = dataloaders.base.__dict__[args.dataset]

    # SPLIT CUB
    if args.is_split_cub :
        print("running split -------------")
        from dataloaders.cub import CUB
        Dataset = CUB
        if args.train_aug :
            print("train aug not supported for cub")
            return
        train_dataset, val_dataset = Dataset(args.dataroot)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)
        n_tasks = len(task_output_space.items())
    # Permuted MNIST
    elif args.n_permutation > 0:
        # TODO : CHECK subset_size
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug, angle=0, subset_size=args.subset_size)
        print("Working with permuatations :) ")
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class=not args.no_class_remap)
        n_tasks = args.n_permutation
    # Rotated MNIST
    elif args.n_rotate > 0 or len(args.rotations) > 0 :
        # TODO : Check subset size
        train_dataset_splits, val_dataset_splits, task_output_space = RotatedGen(Dataset=Dataset,
                                                                                 dataroot=args.dataroot,
                                                                                 train_aug=args.train_aug,
                                                                                 n_rotate=args.n_rotate,
                                                                                 rotate_step=args.rotate_step,
                                                                                 remap_class=not args.no_class_remap,
                                                                                 rotations=args.rotations,
                                                                                 subset_size=args.subset_size)
        n_tasks = len(task_output_space.items())

    # Split MNIST
    else:
        print("running split -------------")
        # TODO : Check subset size
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug,
                                             angle=0, subset_size=args.subset_size)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)
        n_tasks = len(task_output_space.items())

    print(f"task_output_space {task_output_space}")

    return task_output_space, n_tasks, train_dataset_splits, val_dataset_splits


def run(args, wandb_run, task_output_space, n_tasks, train_dataset_splits, val_dataset_splits):
    # Prepare the Agent (model)
    agent_config = args
    agent_config.out_dim = {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space
    
    val_loaders = [torch.utils.data.DataLoader(val_dataset_splits[str(task_id)],
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers)
                   for task_id in range(1, n_tasks + 1)]
    
    # agent_type : regularisation / ogd_plus / agem
    # agent_name : EWC / SI / MAS
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config,
                                                                       val_loaders=val_loaders,
                                                                       wandb_run=wandb_run)
    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)
    
    if args.offline_training:  # Multi-task learning
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

    else:  # Continual learning
        # Compute the validation scores
        val_accs = [agent.validation(loader) for loader in val_loaders]
        print(f"val_accs : {val_accs} ")
        wandb.run.summary.update({f"R_-1": val_accs})

        for i in tqdm(range(len(task_names)), "task"):
            task_name = task_names[i]
            print(f'====================== Task {task_name} =======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
                                                       batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)
            # TODO : Add this part ASAP
            # if args.incremental_class:
            #     agent.add_valid_output_dim(task_output_space[task_name])

            # Train the agent
            agent.learn_batch(train_loader, val_loader)

            val_accs = [agent.validation(loader) for loader in val_loaders]
            print(f"val_accs : {val_accs} ")
            wandb.run.summary.update({f"R_{i}": val_accs})

    return agent


if __name__ == '__main__':
    class Config(Tap):
        run_name: str
        group_id: str

        gpu: bool = True
        workers: int = 4

        # repeat : int = 5
        start_seed : int = 0
        end_seed : int = 5
        run_seed : int = 0
        val_size: int = 256
        lr: float = 1e-3
        scheduler : bool = False
        nepoch: int = 5
        val_check_interval: int = 300
        batch_size: int = 256
        train_percent_check: float = 1.

        # 1 layer is either a weight layer of biais layer, the count starts from the out side of the DNN
        # For multihead DNNs the heads are not taken into account in the count, the count starts from the layers below
        ogd_start_layer : int = 0
        ogd_end_layer : int = 1e6

        memory_size: int = 100
        hidden_dim: int = 100
        pca: bool = False
        subset_size : float = None

        # AGEM
        agem_mem_batch_size : int = 256
        no_transfer : bool = False

        n_permutation : int = 0
        n_rotate : int = 0
        rotate_step : int = 0
        is_split : bool = False
        data_seed : int = 2
        rotations : List = []

        toy: bool = False
        ogd: bool = False
        ogd_plus: bool = False

        no_random_name : bool = False

        project : str = "iclr-2021-cl-prod"
        wandb_dryrun : bool = False
        wandb_dir : str = SLURM_TMPDIR
        # wandb_dir : str = "/scratch/thang/iclr-2021/wandb-offline"
        dataroot : str = os.path.join(SLURM_TMPDIR, "datasets")

        is_split_cub : bool = False

        # Regularisation methods
        reg_coef : float = 0.

        agent_type : str = "ogd_plus"
        agent_name : str = "OGD"
        model_name : str = "MLP"
        model_type : str = "mlp"

        # Stable SGD
        dropout : float = 0.
        gamma : float = 1.
        is_stable_sgd : bool = False


        # Other :
        momentum : float = 0.
        weight_decay : float = 0.
        print_freq : float = 100

        no_val : bool = False

        def add_arguments(self):
            self.add_argument('--gpuid', nargs="+", type=int, default=[0],
                              help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)
            self.add_argument('--force_out_dim', type=int, default=2,
                              help="Set 0 to let the task decide the required output dimension", required=False)
            self.add_argument('--optimizer', type=str, default='SGD',
                              help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...", required=False)
            self.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100", required=False)
            self.add_argument('--first_split_size', type=int, default=2, required=False)
            self.add_argument('--other_split_size', type=int, default=2, required=False)
            self.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                              help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,"
                                   "6 ...] -> [0,1,2 ...]", required=False)
            self.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                              help="Allow data augmentation during training", required=False)
            self.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                              help="Randomize the classes in splits", required=False)
            self.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                              help="Randomize the order of splits", required=False)
            self.add_argument('--schedule', nargs="+", type=int, default=[5],
                              help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number "
                                   "is the end epoch", required=False)
            self.add_argument('--model_weights', type=str, default=None,
                              help="The path to the file for the model weights (*.pth).", required=False)
            self.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                              help="Force the evaluation on train set", required=False)
            self.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                              help="Non-incremental learning by make all data available in one batch. For measuring "
                                   "the upperbound performance.", required=False)
            self.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                              help="The number of output node in the single-headed model increases along with new "
                                   "categories.", required=False)

    # TODO : check known only
    config = Config().parse_args()
    config = config.as_dict()
    config = dotdict(config)

    if torch.cuda.device_count() == 0 :
        config.gpu = False

    if not config.no_random_name :
        config.group_id = config.group_id
        
    if config.wandb_dryrun :
        os.environ["WANDB_MODE"] = "dryrun"

    torch.manual_seed(config.data_seed)
    np.random.seed(config.data_seed)
    task_output_space, n_tasks, train_dataset_splits, val_dataset_splits = prepare_dataloaders(config)

    print("run started !")

    name = f"{config.run_name}-{config.run_seed}"
    wandb_run = wandb.init(tags=["lightning"],
                           project=config.project,
                           sync_tensorboard=False,
                           group=config.group_id,
                           config=config,
                           job_type="eval",
                           name=name,
                           reinit=True,
                           dir=config.wandb_dir)
    # wandb_run = None
    torch.manual_seed(config.run_seed)
    np.random.seed(config.run_seed)

    start = time.time()

    agent = run(args=config,
        wandb_run=wandb_run,
        task_output_space=task_output_space,
        n_tasks=n_tasks,
        train_dataset_splits=train_dataset_splits,
        val_dataset_splits=val_dataset_splits)

    end = time.time()
    elapsed = end - start
    elapsed = time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    wandb.run.summary.update({f"time": elapsed})