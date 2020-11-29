def gen_split_cub_commands():
    nepoch = 75
    memory_size = 100
    start_seed = 0
    end_seed = 5
    subset_size = 20
    hidden_dim = 256

    ogd_tags = ["", "--ogd", "--ogd_plus"] + [" "] * 3
    agent_names = ["OGD"] * 3 + ["EWC", "SI", "MAS"]
    agent_types = ["ogd_plus"] * 3 + ["regularization"] * 3
    algo_names = ["sgd", "ogd", "ogd-plus", "ewc", "si", 'mas']
    lr_list = [0.001] * 6
    batch_size_list = [10] * 6
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_tiny_gpu.sh"] * 6
    run_key_list = ["python main.py"] * 6
    reg_coef_list = [0.] * 3 + [1000, 10, 1]

    for ogd_tag, agent_name, agent_type, algo_name,\
        run_key, reg_coef, batch_size, lr in zip(ogd_tags, agent_names, agent_types, algo_names,
                                             run_key_list, reg_coef_list, batch_size_list, lr_list):
        name = f"split-cub.{algo_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name} --run_name {name} " \
                      f"--is_split --force_out_dim 0 --first_split_size {subset_size} " \
                      f"--other_split_size {subset_size} --hidden_dim {hidden_dim} --model_type alexnet " \
                      f" --nepoch {nepoch} --model_name AlexNetCL --lr {lr}  --reg_coef {reg_coef}" \
                      f" --batch_size {batch_size} --run_seed {run_seed} {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type} --no_val --is_split_cub --wandb_dryrun"
            command_list.append(command)
        command_list.append("\n")
    return command_list

def gen_split_mnist_commands():
    batch_size = 32
    nepoch = 5
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    agem_mem_batch_size = 256
    start_seed = 0
    end_seed = 5
    lr = 0.001

    ogd_tags = ["", "--ogd", "--ogd_plus"] + [" "] * 3
    agent_names = ["OGD"] * 3 + ["EWC", "SI", "MAS"]
    agent_types = ["ogd_plus"] * 3 + ["regularization"] * 3
    algo_names = ["sgd", "ogd", "ogd-plus", "ewc", "si", 'mas']
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_small.sh"] * 6
    run_key_list = ["python main.py"] * 6
    reg_coef_list = [0.] * 3 + [1000, 10000, 10]

    for ogd_tag, agent_name, agent_type, algo_name, run_key, reg_coef in zip(ogd_tags, agent_names, agent_types,
                                                                   algo_names, run_key_list, reg_coef_list):
        name = f"split-mnist.{algo_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name}  --dataset MNIST " \
                      f"--run_name {name} --is_split --force_out_dim 0 --first_split_size 2 " \
                      f"--other_split_size 2 --rand_split  --nepoch {nepoch} --batch_size {batch_size} --run_seed " \
                      f"{run_seed}" \
                      f" --hidden_dim {hidden_dim} --val_check_interval {val_check_interval} {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type}  --no_val --wandb_dryrun " \
                      f"  --reg_coef {reg_coef}"
            command_list.append(command)
        command_list.append("\n")
    return command_list


def gen_rotated_mnist_commands():
    n_rotate = 15
    rotate_step = 5
    batch_size = 32
    nepoch = 5
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    start_seed = 0
    end_seed = 5
    lr = 0.001

    reg_coef_list = [0.] * 3 + [100, 10, 10]
    ogd_tags = ["", "--ogd", "--ogd_plus"] + [" "] * 3
    agent_names = ["OGD"] * 3 + ["EWC", "SI", "MAS"]
    agent_types = ["ogd_plus"] * 3 + ["regularization"] * 3
    algo_names = ["sgd", "ogd", "ogd-plus", "ewc", "si", 'mas']
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_small.sh"] + ["sbatch launch_med_gpu.sh"] * 2 + ["sbatch launch_small.sh"] * 3
    run_key_list = ["python main.py"] * 6

    for ogd_tag, agent_name, agent_type, algo_name, run_key, reg_coef in zip(ogd_tags, agent_names, agent_types,
                                                                   algo_names, run_key_list, reg_coef_list):
        name = f"rotated-mnist.{algo_name}.$group_tag"
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


def gen_permuted_mnist_commands():
    n_permutation = 15
    batch_size = 32
    nepoch = 5
    memory_size = 100
    hidden_dim = 100
    val_check_interval = 10000000
    agem_mem_batch_size = 256
    start_seed = 0
    end_seed = 5
    lr = 0.001

    ogd_tags = ["", "--ogd", "--ogd_plus"] + [" "] * 3
    agent_names = ["OGD"] * 3 + ["EWC", "SI", "MAS"]
    agent_types = ["ogd_plus"] * 3 + ["regularization"] * 3
    algo_names = ["sgd", "ogd", "ogd-plus", "ewc", "si", 'mas']
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_small.sh"] + ["sbatch launch_med_gpu.sh"] * 2 + ["sbatch launch_small.sh"] * 3
    run_key_list = ["python main.py"] * 6
    # TODO : ADD reg coeff
    reg_coef_list = [0.] * 3 + [1, 10, 1]

    for ogd_tag, agent_name, agent_type, algo_name, run_key, reg_coef in zip(ogd_tags, agent_names, agent_types,
                                                                   algo_names, run_key_list, reg_coef_list):
        name = f"permuted-mnist.{algo_name}.$group_tag"
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


def gen_split_cifar_commands():
    batch_size = 32
    # FIXME
    nepoch = 50
    memory_size = 100
    hidden_dim = 100
    start_seed = 0
    end_seed = 5
    lr = 0.001
    subset_size = 5
    reg_coef_list = [0.] * 3 + [10, 10, 10]

    ogd_tags = ["", "--ogd", "--ogd_plus"] + [" "] * 3
    agent_names = ["OGD"] * 3 + ["EWC", "SI", "MAS"]
    agent_types = ["ogd_plus"] * 3 + ["regularization"] * 3
    algo_names = ["sgd", "ogd", "ogd-plus", "ewc", "si", 'mas']
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    # run_key_list = ["sbatch launch_tiny_gpu.sh"] * 3  + ["sbatch launch_large_gpu.sh"] * 3
    run_key_list = ["python main.py"] * 6
    
    for ogd_tag, agent_name, agent_type, algo_name, run_key, reg_coef in zip(ogd_tags, agent_names, agent_types,
                                                                   algo_names, run_key_list, reg_coef_list):
        name = f"split-cifar100.{algo_name}.$group_tag"
        for run_seed in range(start_seed, end_seed):
            command = f"{run_key} --memory_size {memory_size} --group_id {name}  --dataset CIFAR100 --run_name {name} " \
                      f"" \
                      f"--is_split --force_out_dim 0 --first_split_size {subset_size} --other_s" \
                      f"plit_size {subset_size} " \
                      f"--hidden_dim {hidden_dim} --nepoch {nepoch} --model_type lenet --model_name LeNetC --lr {lr} " \
                      f" --batch_size {batch_size} --run_seed {run_seed}  {ogd_tag} " \
                      f"--agent_name {agent_name} --agent_type {agent_type} --no_val --wandb_dryrun " \
                      f" --reg_coef {reg_coef}"
            command_list.append(command)
        command_list.append("\n")
    return command_list


if __name__ == '__main__':
    commands_dict = dict(
        prod_split_mnist=gen_split_mnist_commands(),
        prod_split_cifar=gen_split_cifar_commands(),
        prod_permuted_mnist=gen_permuted_mnist_commands(),
        prod_rotated_mnist=gen_rotated_mnist_commands(),
        prod_split_cub=gen_split_cub_commands()
    )
    for key, command_list in commands_dict.items():
        print(f"{key} has {len(command_list)} commands")
        with open(f"commands/{key}.sh", "w") as outfile:
            outfile.write("\n".join(command_list))