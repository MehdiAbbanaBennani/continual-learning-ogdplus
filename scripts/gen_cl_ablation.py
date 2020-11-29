def gen_command(command_dict, prefix_list):
    command_list = prefix_list.copy()
    for key, val in command_dict.items():
        if isinstance(val, bool):
            if val == True:
                command_list.append(f"--{key}")
        else:
            command_list.append(f"--{key} {val}")
    return " ".join(command_list)


def gen_permutations(grid_dict, prefix_list, header_list):
    for key, val in grid_dict.items():
        if not isinstance(val, list):
            grid_dict[key] = [val]
    
    import itertools
    keys, values = zip(*grid_dict.items())
    dicts_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    commands_list = [gen_command(command_dict=command_dict, prefix_list=prefix_list) for command_dict in dicts_list]
    commands_list = header_list + commands_list
    return commands_list


name = f"permuted-mnist.ablation_size.$group_tag"
ablation_permuted_size = {
    "n_permutation": 15,
    "no_class_remap": True,
    "memory_size": 100,
    "group_id": name,
    "dataset": "MNIST",
    "run_name": name,
    "force_out_dim": 10,
    "run_seed": [i for i in range(5)],
    "lr": 0.001,
    "batch_size": 32,
    "hidden_dim": 100,
    "nepoch": 5,
    "agent_name": "OGD",
    "agent_type": "ogd_plus",
    "no_val": True,
    "wandb_dryrun": True,
    "subset_size": [0.25, 0.5, 0.75, 1],
    "ogd": True
}

name = f"rotated-mnist.ablation_size.$group_tag"
ablation_rotated_size = {
    "force_out_dim": 10,
    "no_class_remap": True,
    "group_id": name,
    "dataset": "MNIST",
    "n_rotate": 15,
    "rotate_step": 5,
    "run_name": name,
    "run_seed": [i for i in range(5)],
    "memory_size": 100,
    "lr": 0.001,
    "batch_size": 32,
    "hidden_dim": 100,
    "nepoch": 5,
    "agent_name": "OGD",
    "agent_type": "ogd_plus",
    "no_val": True,
    "wandb_dryrun": True,
    "ogd": True,
    "subset_size": [0.25, 0.5, 0.75, 1]
}

name = f"split-cifar100.ablation_size.$group_tag"
ablation_cifar100_size = {
    "memory_size": 100,
    "group_id": name,
    "dataset": "CIFAR100",
    "run_name": name,
    "is_split": True,
    "force_out_dim": 0,
    "first_split_size": 5,
    "other_split_size": 5,
    "hidden_dim": 100,
    "nepoch": 50,
    "model_type": "lenet",
    "model_name": "LeNetC",
    "lr": 0.001,
    "batch_size": 32,
    "run_seed": [i for i in range(5)],
    "rand_split": True,
    "agent_name": "OGD",
    "agent_type": "ogd_plus",
    "no_val": True,
    "wandb_dryrun": True,
    "ogd": True,
    "subset_size": [0.5, 0.75, 1]
}

if __name__ == '__main__':
    header_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n", ]
    prefix_list = ["python main.py"]
    # prefix_list = ["sbatch launch_small.sh"]
    
    commands_dict = dict(
        ablation_permuted_size=gen_permutations(grid_dict=ablation_permuted_size,
                                                prefix_list=prefix_list,
                                                header_list=header_list),
        ablation_rotated_size=gen_permutations(grid_dict=ablation_rotated_size,
                                               prefix_list=prefix_list,
                                               header_list=header_list),
        ablation_cifar100_size=gen_permutations(grid_dict=ablation_cifar100_size,
                                                prefix_list=prefix_list,
                                                header_list=header_list),
    )
    
    for key, command_list in commands_dict.items():
        print(f"{key} has {len(command_list)} commands")
        with open(f"commands/{key}.sh", "w") as outfile:
            outfile.write("\n".join(command_list))
