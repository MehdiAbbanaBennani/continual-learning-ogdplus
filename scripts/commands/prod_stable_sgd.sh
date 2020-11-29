group_tag=$(date +'%Y-%m-%d-%H-%M')


sbatch launch_tiny_stable.sh --dataset cifar100 --tasks 20 --epochs-per-task 50 --lr 0.01 --gamma 0.8 --hiddens 100 --batch-size 10 --dropout 0.4 --seed 0 --group_id split-cifar100.stable_sgd.$group_tag --run_name split-cifar100.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset rot-mnist --tasks 15 --epochs-per-task 1  --lr 0.25 --gamma 0.5 --hiddens 100 --batch-size 64 --dropout 0.1 --seed 0 --group_id rotated-mnist.stable_sgd.$group_tag --run_name rotated-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset perm-mnist --tasks 15 --epochs-per-task 1  --lr 0.01 --gamma 0.6 --hiddens 100 --batch-size 10 --dropout 0.2  --seed 0 --group_id permuted-mnist.stable_sgd.$group_tag --run_name permuted-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset cub200 --tasks 10 --epochs-per-task 75 --lr 0.01 --gamma 0.6 --hiddens 256 --batch-size 10 --dropout 0.4 --seed 0 --group_id split-cub.stable_sgd.$group_tag --run_name split-cub.stable_sgd.$group_tag --wandb_dryrun


sbatch launch_tiny_stable.sh --dataset cifar100 --tasks 20 --epochs-per-task 50 --lr 0.01 --gamma 0.8 --hiddens 100 --batch-size 10 --dropout 0.4 --seed 1 --group_id split-cifar100.stable_sgd.$group_tag --run_name split-cifar100.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset rot-mnist --tasks 15 --epochs-per-task 1  --lr 0.25 --gamma 0.5 --hiddens 100 --batch-size 64 --dropout 0.1 --seed 1 --group_id rotated-mnist.stable_sgd.$group_tag --run_name rotated-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset perm-mnist --tasks 15 --epochs-per-task 1  --lr 0.01 --gamma 0.6 --hiddens 100 --batch-size 10 --dropout 0.2  --seed 1 --group_id permuted-mnist.stable_sgd.$group_tag --run_name permuted-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset cub200 --tasks 10 --epochs-per-task 75 --lr 0.01 --gamma 0.6 --hiddens 256 --batch-size 10 --dropout 0.4 --seed 1 --group_id split-cub.stable_sgd.$group_tag --run_name split-cub.stable_sgd.$group_tag --wandb_dryrun


sbatch launch_tiny_stable.sh --dataset cifar100 --tasks 20 --epochs-per-task 50 --lr 0.01 --gamma 0.8 --hiddens 100 --batch-size 10 --dropout 0.4 --seed 2 --group_id split-cifar100.stable_sgd.$group_tag --run_name split-cifar100.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset rot-mnist --tasks 15 --epochs-per-task 1  --lr 0.25 --gamma 0.5 --hiddens 100 --batch-size 64 --dropout 0.1 --seed 2 --group_id rotated-mnist.stable_sgd.$group_tag --run_name rotated-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset perm-mnist --tasks 15 --epochs-per-task 1  --lr 0.01 --gamma 0.6 --hiddens 100 --batch-size 10 --dropout 0.2  --seed 2 --group_id permuted-mnist.stable_sgd.$group_tag --run_name permuted-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset cub200 --tasks 10 --epochs-per-task 75 --lr 0.01 --gamma 0.6 --hiddens 256 --batch-size 10 --dropout 0.4 --seed 2 --group_id split-cub.stable_sgd.$group_tag --run_name split-cub.stable_sgd.$group_tag --wandb_dryrun


sbatch launch_tiny_stable.sh --dataset cifar100 --tasks 20 --epochs-per-task 50 --lr 0.01 --gamma 0.8 --hiddens 100 --batch-size 10 --dropout 0.4 --seed 3 --group_id split-cifar100.stable_sgd.$group_tag --run_name split-cifar100.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset rot-mnist --tasks 15 --epochs-per-task 1  --lr 0.25 --gamma 0.5 --hiddens 100 --batch-size 64 --dropout 0.1 --seed 3 --group_id rotated-mnist.stable_sgd.$group_tag --run_name rotated-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset perm-mnist --tasks 15 --epochs-per-task 1  --lr 0.01 --gamma 0.6 --hiddens 100 --batch-size 10 --dropout 0.2  --seed 3 --group_id permuted-mnist.stable_sgd.$group_tag --run_name permuted-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset cub200 --tasks 10 --epochs-per-task 75 --lr 0.01 --gamma 0.6 --hiddens 256 --batch-size 10 --dropout 0.4 --seed 3 --group_id split-cub.stable_sgd.$group_tag --run_name split-cub.stable_sgd.$group_tag --wandb_dryrun


sbatch launch_tiny_stable.sh --dataset cifar100 --tasks 20 --epochs-per-task 50 --lr 0.01 --gamma 0.8 --hiddens 100 --batch-size 10 --dropout 0.4 --seed 4 --group_id split-cifar100.stable_sgd.$group_tag --run_name split-cifar100.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset rot-mnist --tasks 15 --epochs-per-task 1  --lr 0.25 --gamma 0.5 --hiddens 100 --batch-size 64 --dropout 0.1 --seed 4 --group_id rotated-mnist.stable_sgd.$group_tag --run_name rotated-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset perm-mnist --tasks 15 --epochs-per-task 1  --lr 0.01 --gamma 0.6 --hiddens 100 --batch-size 10 --dropout 0.2  --seed 4 --group_id permuted-mnist.stable_sgd.$group_tag --run_name permuted-mnist.stable_sgd.$group_tag --wandb_dryrun
sbatch launch_tiny_stable.sh --dataset cub200 --tasks 10 --epochs-per-task 75 --lr 0.01 --gamma 0.6 --hiddens 256 --batch-size 10 --dropout 0.4 --seed 4 --group_id split-cub.stable_sgd.$group_tag --run_name split-cub.stable_sgd.$group_tag --wandb_dryrun

