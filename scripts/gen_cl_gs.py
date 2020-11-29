import itertools


def gen_gs_split_cub_commands_non_ogd():
    nepoch = 75
    memory_size = 100
    start_seed = 0
    end_seed = 1
    subset_size = 20
    hidden_dim = 256
    
    ogd_tag = " "
    # run_key = "sbatch launch_med_gpu.sh"
    run_key = "python main.py"
    agent_names = ["EWC", "MAS", "SI"]
    agent_types = ["regularization"]
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    reg_coef_list = [10 ** x for x in range(-1, 6)]
    lr_list = [0.001, 0.01, 0.1]
    batch_size_list = [10, 64, 256]
    
    for agent_name, agent_type, reg_coef, lr, batch_size in itertools.product(agent_names,
                                                                              agent_types,
                                                                              reg_coef_list,
                                                                              lr_list,
                                                                              batch_size_list):
        name = f"gs_split-cub.{agent_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name} --run_name {name} " \
                      f"--is_split --force_out_dim 0 --first_split_size {subset_size} " \
                      f"--other_split_size {subset_size} --hidden_dim {hidden_dim} --model_type alexnet " \
                      f" --nepoch {nepoch} --model_name AlexNetCL --lr {lr}  --reg_coef {reg_coef}" \
                      f" --batch_size {batch_size} --run_seed {run_seed}   {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type} --no_val --is_split_cub --wandb_dryrun"
            command_list.append(command)
        command_list.append("\n")
    return command_list


def gen_gs_split_cub_commands_ogd():
    nepoch = 75
    memory_size = 100
    start_seed = 0
    end_seed = 1
    subset_size = 20
    hidden_dim = 256
    
    # run_key = "sbatch launch_med_gpu.sh"
    run_key = "python main.py"

    ogd_tag_list = ["", "--ogd", "--ogd_plus"]
    agent_name = "OGD"
    agent_type = "ogd_plus"
    reg_coef = 0
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    lr_list = [0.001, 0.01, 0.1]
    batch_size_list = [10, 64, 256]
    
    algo_name_map = {"": "sgd",
                     "--ogd": "ogd",
                     "--ogd_plus": "ogd-plus"}
    
    for ogd_tag, lr, batch_size in itertools.product(ogd_tag_list, lr_list, batch_size_list):
        algo_name = algo_name_map[ogd_tag]
        name = f"gs_split-cub.{algo_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name} --run_name {name} " \
                      f"--is_split --force_out_dim 0 --first_split_size {subset_size} " \
                      f"--other_split_size {subset_size} --hidden_dim {hidden_dim} --model_type alexnet " \
                      f" --nepoch {nepoch} --model_name AlexNetCL --lr {lr}  --reg_coef {reg_coef}" \
                      f" --batch_size {batch_size} --run_seed {run_seed}   {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type} --no_val --is_split_cub --wandb_dryrun"
            command_list.append(command)
        command_list.append("\n")
    return command_list


def gen_gs_split_mnist_commands(agent):
    batch_size = 32
    nepoch = 5
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    start_seed = 0
    end_seed = 1
    lr = 0.001
    
    n_config = 8
    ogd_tags = [" "] * n_config
    agent_names = [agent] * n_config
    agent_types = ["regularization"] * n_config
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_tiny.sh"] * n_config
    run_key_list = ["python main.py"] * n_config
    reg_coef_list = [10 ** x for x in range(-1, 7)]
    
    for ogd_tag, agent_name, agent_type, run_key, reg_coef in zip(ogd_tags,
                                                                  agent_names,
                                                                  agent_types,
                                                                  run_key_list,
                                                                  reg_coef_list):
        str_reg = str(reg_coef).replace(".", ",")
        name = f"gs_split-mnist_{agent_name}.{str_reg}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name}  --dataset MNIST " \
                      f"--run_name {name} --is_split --force_out_dim 0 --dataset MNIST --first_split_size 2 " \
                      f"--other_split_size 2   --nepoch {nepoch} --batch_size {batch_size} --run_seed " \
                      f"{run_seed}" \
                      f" --hidden_dim {hidden_dim} --val_check_interval {val_check_interval} {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type}  --no_val --wandb_dryrun " \
                      f" --reg_coef {reg_coef} "
            command_list.append(command)
        command_list.append("\n")
    return command_list


def gen_gs_rotated_mnist_commands():
    n_rotate = 15
    rotate_step = 5
    batch_size = 32
    nepoch = 5
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    start_seed = 0
    end_seed = 1
    lr = 0.001
    
    n_config = 8
    ogd_tags = [" "] * n_config
    agent_types = ["regularization"] * n_config
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_small.sh"] * n_config
    run_key_list = ["python main.py"] * n_config
    agent_names = ["EWC", "MAS", "SI"]
    reg_coef_list = [10 ** x for x in range(-1, 7)]
    
    for agent_name in agent_names :
        for ogd_tag, agent_type, run_key, reg_coef in zip(ogd_tags,
                                                                      agent_types,
                                                                      run_key_list,
                                                                      reg_coef_list):
            
            str_reg = str(reg_coef).replace(".", ",")
            name = f"gs_rotated-mnist_{agent_name}.{str_reg}.$group_tag"
            for run_seed in range(start_seed, end_seed):
                command = f"{run_key} --force_out_dim 10 --no_class_remap --group_id {name} --dataset MNIST " \
                          f"--n_rotate {n_rotate} --rotate_step {rotate_step} --run_name {name} --run_seed {run_seed} " \
                          f"--memory_size {memory_size} --lr {lr} --batch_size {batch_size} --val_check_interval " \
                          f"{val_check_interval} --hidden_dim {hidden_dim} --nepoch {nepoch} {ogd_tag} " \
                          f"--agent_name {agent_name} --agent_type {agent_type} " \
                          f" --no_val --wandb_dryrun --reg_coef {reg_coef}"
                
                command_list.append(command)
            command_list.append("\n")
    return command_list


def gen_gs_permuted_mnist_commands():
    # TODO : Change
    n_permutation = 15
    batch_size = 32
    nepoch = 5
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    agem_mem_batch_size = 256
    start_seed = 0
    end_seed = 1
    lr = 0.001
    
    n_config = 8
    ogd_tags = [" "] * n_config
    agent_names = ["EWC", "MAS", "SI"]
    agent_types = ["regularization"] * n_config
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_small.sh"] * n_config
    run_key_list = ["python main.py"] * n_config
    reg_coef_list = [10 ** x for x in range(-1, 7)]
    
    for agent_name in agent_names :
        for ogd_tag, agent_type, run_key, reg_coef in zip(ogd_tags,
                                                                      agent_types,
                                                                      run_key_list,
                                                                      reg_coef_list):
            str_reg = str(reg_coef).replace(".", ",")
            name = f"gs_permuted-mnist_{agent_name}.{str_reg}.$group_tag"
            for run_seed in range(start_seed, end_seed):
                command = f"{run_key} --n_permutation {n_permutation} --force_out_dim 10 --no_class_remap  --dataset " \
                          f"MNIST " \
                          f"--run_name {name} --group_id {name} --run_seed {run_seed} --memory_size {memory_size} " \
                          f"--lr {lr} --batch_size {batch_size} --val_check_interval {val_check_interval} --hidden_dim " \
                          f"{hidden_dim} --nepoch {nepoch} {ogd_tag} --agent_name {agent_name} --agent_type {agent_type}" \
                          f" --no_val --wandb_dryrun --reg_coef {reg_coef}"
                command_list.append(command)
            command_list.append("\n")
    return command_list


def gen_gs_split_cifar_non_ogd_commands():
    nepoch = 50
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    start_seed = 0
    end_seed = 1
    subset_size = 5
    
    ogd_tag = " "
    agent_names = ["EWC", "MAS", "SI"]
    agent_type = "regularization"
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key = "sbatch launch_large_gpu.sh"
    run_key = "python main.py"
    reg_coef_list = [10 ** x for x in range(-1, 5)]
    lr_list = [0.00001, 0.001, 0.01, 0.1]
    batch_size_list = [32, 64, 256]
    
    for lr, agent_name, batch_size, reg_coef in itertools.product(lr_list,
                                                                  agent_names,
                                                                  batch_size_list,
                                                                  reg_coef_list):
        # if agent_name == "EWC" :
        name = f"gs_split-cifar100.{agent_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name}  --dataset CIFAR100 --run_name {name} " \
                      f"--is_split --force_out_dim 0 --first_split_size {subset_size} " \
                      f"--other_split_size {subset_size} " \
                      f"--hidden_dim {hidden_dim} --nepoch {nepoch} --model_type lenet --model_name LeNetC --lr {lr} " \
                      f" --batch_size {batch_size} --run_seed {run_seed}  {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type} --no_val --wandb_dryrun " \
                      f" --reg_coef {reg_coef} --val_check_interval {val_check_interval}"
            command_list.append(command)
            # command_list.append("\n")
    return command_list


def gen_gs_split_cifar_ogd_commands():
    import itertools
    
    memory_size = 100
    start_seed = 0
    end_seed = 1
    subset_size = 5
    reg_coef = 0.
    algo_name_map = {"": "sgd",
                     "--ogd": "ogd",
                     "--ogd_plus": "ogd-plus"}
    ogd_tag_list = ["", "--ogd", "--ogd_plus"]
    lr_list = [0.00001, 0.001, 0.01, 0.1]
    batch_size_list = [32, 64, 256]
    hidden_list = [100, 256]
    nepoch_list = [1, 20, 50]
    
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    
    agent_name = "OGD"
    agent_type = "ogd_plus"
    # run_key = "sbatch launch_tiny_gpu.sh"
    run_key = "python main.py"
    
    for ogd_tag, lr, batch_size, hidden_dim, nepoch in itertools.product(ogd_tag_list, lr_list,
                                                                         batch_size_list, hidden_list, nepoch_list):
        algo_name = algo_name_map[ogd_tag]
        name = f"gs_split-cifar100.{algo_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name}  --dataset CIFAR100 --run_name {name} " \
                      f"" \
                      f"--is_split --force_out_dim 0 --first_split_size {subset_size} --other_s" \
                      f"plit_size {subset_size} " \
                      f"--hidden_dim {hidden_dim} --nepoch {nepoch} --model_type lenet --model_name LeNetC --lr {lr} " \
                      f" --batch_size {batch_size} --run_seed {run_seed}  {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type} --no_val " \
                      f" --reg_coef {reg_coef} --wandb_dryrun "
            command_list.append(command)
    return command_list


if __name__ == '__main__':
    commands_list = list()
    
    commands_dict = dict(
        gs_cifar_non_ogd=gen_gs_split_cifar_non_ogd_commands(),
        gs_cifar_ogd=gen_gs_split_cifar_ogd_commands(),
        gs_cub_non_ogd=gen_gs_split_cub_commands_non_ogd(),
        gs_cub_ogd=gen_gs_split_cub_commands_ogd(),
        gs_rotated_mnist_non_ogd=gen_gs_rotated_mnist_commands(),
        gs_permuted_mnist_non_ogd=gen_gs_permuted_mnist_commands(),
        
    )
    for key, command_list in commands_dict.items():
        print(f"{key} has {len(command_list)} commands")
        with open(f"commands/{key}.sh", "w") as outfile:
            outfile.write("\n".join(command_list))
