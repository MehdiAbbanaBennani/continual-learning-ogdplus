import itertools


def gen_gs_commands():
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    
    seed_list = [0]
    gamma_list = [0.1 * i for i in range(5, 10)]
    lr_list = [0.01, 0.1, 0.25]
    batch_size_list = [10, 32, 64]
    dropout_list = [0.1 * i for i in range(0, 6)]
    # run_key = "sbatch launch_tiny_stable.sh"
    run_key = "python -m stable_sgd.main"

    for seed, gamma, lr, dropout, batch_size in itertools.product(seed_list,
                                                                  gamma_list,
                                                                  lr_list,
                                                                  dropout_list,
                                                                  batch_size_list):
        name = f"rotated-mnist.stable_sgd.$group_tag"
        run_name = name
        rotated_command = f"{run_key} --dataset rot-mnist --tasks 15 --epochs-per-task 1 " \
                          f" --lr {lr} --gamma {gamma} --hiddens 100 --batch-size {batch_size} --dropout {dropout} " \
                          f"--seed {seed} --group_id {name} --run_name {run_name} --wandb_dryrun"
        command_list.append(rotated_command)

        name = f"permuted-mnist.stable_sgd.$group_tag"
        run_name = name
        permuted_command = f"{run_key} --dataset perm-mnist --tasks 15 --epochs-per-task 1 " \
                           f" --lr {lr} --gamma {gamma} --hiddens 100 --batch-size {batch_size} --dropout {dropout} " \
                           f" --seed {seed} --group_id {name} --run_name {run_name} --wandb_dryrun"
        command_list.append(permuted_command)

    seed_list = [0]
    gamma_list = [0.1 * i for i in range(5, 9)]
    lr_list = [0.001, 0.01, 0.1]
    batch_size_list = [10, 64]
    dropout_list = [0.1 * i for i in range(0, 6)]
    epochs_list = [1, 10, 50]
    run_key = "sbatch launch_tiny_stable.sh"
    for seed, gamma, lr, dropout, batch_size, epochs in itertools.product(seed_list,
                                                                  gamma_list,
                                                                  lr_list,
                                                                  dropout_list,
                                                                  batch_size_list,
                                                                  epochs_list):
        name = f"split-cifar100.stable_sgd.$group_tag"
        run_name = name
        cifar_command = f"{run_key} --dataset cifar100 --tasks 20 --epochs-per-task {epochs} --lr {lr} --gamma {gamma}" \
                        f" --hiddens 100 --batch-size {batch_size} --dropout {dropout} --seed 1 --group_id {name} " \
                        f"--run_name {run_name} --wandb_dryrun"
        command_list.append(cifar_command)

    seed_list = [0]
    gamma_list = [0.1 * i for i in range(5, 11)]
    lr_list = [0.001, 0.01, 0.1]
    batch_size_list = [10, 64]
    dropout_list = [0.1 * i for i in range(0, 6)]
    epochs_list = [10, 75]
    run_key = "sbatch launch_tiny_stable.sh"
    # run_key = "python -m stable_sgd.main"
    for seed, gamma, lr, dropout, batch_size, epochs in itertools.product(seed_list,
                                                                          gamma_list,
                                                                          lr_list,
                                                                          dropout_list,
                                                                          batch_size_list,
                                                                          epochs_list):
        name = f"split-cub.stable_sgd.$group_tag"
        run_name = name
        cifar_command = f"{run_key} --dataset cub200 --tasks 10 --epochs-per-task {epochs} --lr {lr} --gamma {gamma}" \
                        f" --hiddens 256 --batch-size {batch_size} --dropout {dropout} --seed 1 --group_id {name} " \
                        f"--run_name {run_name} " \
                        f"--wandb_dryrun"
        command_list.append(cifar_command)
    
    return command_list


def gen_prod_commands():
    command_list = ["group_tag=$(date +'%Y-%m-%d-%H-%M')", "\n"]
    run_key = "sbatch launch_tiny_stable.sh"
    start_seed = 0
    end_seed = 5
    
    for seed in range(start_seed, end_seed):
        epochs = 50
        lr = 0.01
        gamma = 0.8
        batch_size = 10
        dropout = 0.4
        name = f"split-cifar100.stable_sgd.$group_tag"
        run_name = name
        cifar_command = f"{run_key} --dataset cifar100 --tasks 20 --epochs-per-task {epochs} --lr {lr} --gamma {gamma}" \
                        f" --hiddens 100 --batch-size {batch_size} --dropout {dropout} --seed {seed} --group_id {name} " \
                        f"--run_name {run_name} --wandb_dryrun"
        command_list.append(cifar_command)

        epochs = 1
        lr = 0.25
        gamma = 0.5
        batch_size = 64
        dropout = 0.1
        name = f"rotated-mnist.stable_sgd.$group_tag"
        run_name = name
        rotated_command = f"{run_key} --dataset rot-mnist --tasks 15 --epochs-per-task {epochs} " \
                          f" --lr {lr} --gamma {gamma} --hiddens 100 --batch-size {batch_size} --dropout {dropout} " \
                          f"--seed {seed} --group_id {name} --run_name {run_name} --wandb_dryrun"
        command_list.append(rotated_command)

        epochs = 1
        lr = 0.01
        gamma = 0.6
        batch_size = 10
        dropout = 0.2
        name = f"permuted-mnist.stable_sgd.$group_tag"
        run_name = name
        permuted_command = f"{run_key} --dataset perm-mnist --tasks 15 --epochs-per-task {epochs} " \
                           f" --lr {lr} --gamma {gamma} --hiddens 100 --batch-size {batch_size} --dropout {dropout} " \
                           f" --seed {seed} --group_id {name} --run_name {run_name} --wandb_dryrun"
        command_list.append(permuted_command)


        name = f"split-cub.stable_sgd.$group_tag"
        cub_command = f"{run_key} --dataset cub200 --tasks 10 --epochs-per-task 75 --lr 0.01" \
                      f" --gamma 0.6 --hiddens 256 --batch-size 10 --dropout 0.4 --seed {seed} " \
                      f"--group_id {name} --run_name {name} --wandb_dryrun"
        command_list.append(cub_command)

        command_list.append("\n")
    
    return command_list


if __name__ == '__main__':
    commands_dict = dict(
        prod_stable_sgd=gen_prod_commands(),
        gs_stable_sgd=gen_gs_commands(),
    )
    
    for key, command_list in commands_dict.items():
        print(f"{key} has {len(command_list)} commands")
        with open(f"commands/{key}.sh", "w") as outfile:
            outfile.write("\n".join(command_list))
