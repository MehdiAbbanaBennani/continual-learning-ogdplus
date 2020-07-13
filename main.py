import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import agents

from .utils import dotdict
from tap import Tap
import wandb
import numpy as np
import torch
from random import shuffle


def prepare_dataloaders(args):
    # Prepare dataloaders
    Dataset = dataloaders.base.__dict__[args.dataset]

    # Permuted MNIST
    if args.n_permutation > 0:
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug, angle=0)
        print("Working with permuatations :) ")
        train_dataset_splits, val_dataset_splits, task_output_space = PermutedGen(train_dataset, val_dataset,
                                                                                  args.n_permutation,
                                                                                  remap_class=not args.no_class_remap)
        n_tasks = args.n_permutation
    # Rotated MNIST
    elif args.n_rotate > 0:
        train_dataset_splits, val_dataset_splits, task_output_space = RotatedGen(Dataset=Dataset,
                                                                                 dataroot=args.dataroot,
                                                                                 train_aug=args.train_aug,
                                                                                 n_rotate=args.n_rotate,
                                                                                 rotate_step=args.rotate_step,
                                                                                 remap_class=not args.no_class_remap)
        n_tasks = args.n_rotate

    # Split MNIST
    else:
        print("running split -------------")
        train_dataset, val_dataset = Dataset(args.dataroot, args.train_aug, angle=0)
        train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset, val_dataset,
                                                                               first_split_sz=args.first_split_size,
                                                                               other_split_sz=args.other_split_size,
                                                                               rand_split=args.rand_split,
                                                                               remap_class=not args.no_class_remap)
        n_tasks = len(task_output_space.items())

    print(f"task_output_space {task_output_space}")

    return task_output_space, n_tasks, train_dataset_splits, val_dataset_splits


def run(args, wandb_run, task_output_space, n_tasks, train_dataset_splits, val_dataset_splits):
    # Log images to check the data
    def get_samples(dataset, n_samples=10):
        return [wandb.Image(dataset[i][0], caption=f"img {i}") for i in range(n_samples)]

    wandb.log({task_name: get_samples(dataset) for task_name, dataset in train_dataset_splits.items()},
              commit=False)

    # Prepare the Agent (model)
    agent_config = args
    agent_config.out_dim = {'All': args.force_out_dim} if args.force_out_dim > 0 else task_output_space

    val_loaders = [torch.utils.data.DataLoader(val_dataset_splits[str(task_id)],
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers)
                   for task_id in range(1, n_tasks + 1)]

    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config,
                                                                       val_loaders=val_loaders,
                                                                       wandb_run=wandb_run)
    # print(agent.model)
    # print('#parameter of model:', agent.count_parameter())

    # Decide split ordering
    task_names = sorted(list(task_output_space.keys()), key=int)
    print('Task order:', task_names)
    if args.rand_split_order:
        shuffle(task_names)
        print('Shuffled task order:', task_names)

    if args.offline_training:  # Non-incremental learning / offline_training / measure the upper-bound performance
        train_dataset_all = torch.utils.data.ConcatDataset(train_dataset_splits.values())
        val_dataset_all = torch.utils.data.ConcatDataset(val_dataset_splits.values())
        train_loader = torch.utils.data.DataLoader(train_dataset_all,
                                                   batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        val_loader = torch.utils.data.DataLoader(val_dataset_all,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        agent.learn_batch(train_loader, val_loader)

    else:  # Incremental learning
        for i in range(len(task_names)):
            task_name = task_names[i]
            print('======================', task_name, '=======================')
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[task_name],
                                                       batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[task_name],
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers)

            # if args.incremental_class:
            #     agent.add_valid_output_dim(task_output_space[task_name])

            agent.learn_batch(train_loader, val_loader)


if __name__ == '__main__':
    class Config(Tap):
        run_name: str
        group_id: str

        gpu: bool = True
        workers: int = 16

        # repeat : int = 5
        start_seed: int = 0
        end_seed: int = 5
        val_size: int = 256
        lr: float = 1e-2
        scheduler: bool = False
        nepoch: int = 5
        val_check_interval: int = 50
        batch_size: int = 256
        train_percent_check: float = 1.

        memory_size: int = 1000
        hidden_dim: int = 256

        n_permutation: int = 0
        n_rotate: int = 0
        rotate_step: int = 0
        is_split: bool = False
        data_seed: int = 2

        toy: bool = False
        ogd: bool = False
        ogd_plus: bool = False

        no_random_name: bool = False

        def add_arguments(self):
            self.add_argument('--gpuid', nargs="+", type=int, default=[0],
                              help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)
            self.add_argument('--model_type', type=str, default='mlp',
                              help="The type (mlp|lenet|vgg|resnet) of backbone network", required=False)
            self.add_argument('--model_name', type=str, default='MLP',
                              help="The name of actual model for the backbone", required=False)
            self.add_argument('--force_out_dim', type=int, default=2,
                              help="Set 0 to let the task decide the required output dimension", required=False)
            self.add_argument('--agent_type', type=str, default='ogd_plus', help="The type (filename) of agent",
                              required=False)
            self.add_argument('--agent_name', type=str, default='OGD', help="The class name of agent", required=False)
            self.add_argument('--optimizer', type=str, default='SGD',
                              help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...", required=False)
            self.add_argument('--dataroot', type=str, default='data',
                              help="The root folder of dataset or downloaded data", required=False)
            self.add_argument('--dataset', type=str, default='MNIST', help="MNIST(default)|CIFAR10|CIFAR100",
                              required=False)
            # self.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0",
            # required=False)
            self.add_argument('--first_split_size', type=int, default=2, required=False)
            self.add_argument('--other_split_size', type=int, default=2, required=False)
            # TODO : check --no_class_remap ; not sure ...
            self.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                              help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,"
                                   "6 ...] -> [0,1,2 ...]", required=False)
            self.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                              help="Allow data augmentation during training", required=False)
            self.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                              help="Randomize the classes in splits", required=False)
            self.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                              help="Randomize the order of splits", required=False)
            self.add_argument('--workers', type=int, default=3, help="#Thread for dataloader", required=False)
            # self.add_argument('--batch_size', type=int, default=100)
            # self.add_argument('--lr', type=float, default=0.01, help="Learning rate")
            self.add_argument('--momentum', type=float, default=0, required=False)
            self.add_argument('--weight_decay', type=float, default=0, required=False)
            self.add_argument('--schedule', nargs="+", type=int, default=[5],
                              help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number "
                                   "is the end epoch", required=False)
            self.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration",
                              required=False)
            self.add_argument('--model_weights', type=str, default=None,
                              help="The path to the file for the model weights (*.pth).", required=False)
            self.add_argument('--reg_coef', nargs="+", type=float, default=[0.],
                              help="The coefficient for regularization. Larger means less plasilicity. Give a list "
                                   "for hyperparameter search.", required=False)
            self.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                              help="Force the evaluation on train set", required=False)
            self.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                              help="Non-incremental learning by make all data available in one batch. For measuring "
                                   "the upperbound performance.", required=False)
            self.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                              help="The number of output node in the single-headed model increases along with new "
                                   "categories.", required=False)


    config = Config().parse_args()
    config = config.as_dict()
    config = dotdict(config)

    if not config.no_random_name:
        config.group_id = config.group_id + "-" + wandb.util.generate_id()

    torch.manual_seed(config.data_seed)
    np.random.seed(config.data_seed)
    task_output_space, n_tasks, train_dataset_splits, val_dataset_splits = prepare_dataloaders(config)

    for repeat_id in range(config.start_seed, config.end_seed):
        config.run_seed = repeat_id + 1

        name = f"{config.run_name}-{repeat_id}"
        wandb_run = wandb.init(tags=["lightning"],
                               project="research-continual-learning",
                               sync_tensorboard=False,
                               group=config.group_id,
                               config=config,
                               job_type="train",
                               name=name,
                               reinit=True)
        torch.manual_seed(config.run_seed)
        np.random.seed(config.run_seed)

        run(args=config,
            wandb_run=wandb_run,
            task_output_space=task_output_space,
            n_tasks=n_tasks,
            train_dataset_splits=train_dataset_splits,
            val_dataset_splits=val_dataset_splits)
