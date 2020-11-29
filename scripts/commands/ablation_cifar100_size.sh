group_tag=$(date +'%Y-%m-%d-%H-%M')


python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 0 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.5
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 0 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.75
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 0 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 1
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 1 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.5
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 1 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.75
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 1 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 1
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 2 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.5
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 2 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.75
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 2 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 1
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 3 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.5
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 3 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.75
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 3 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 1
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 4 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.5
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 4 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 0.75
python main.py --memory_size 100 --group_id split-cifar100.ablation_size.$group_tag --dataset CIFAR100 --run_name split-cifar100.ablation_size.$group_tag --is_split --force_out_dim 0 --first_split_size 5 --other_split_size 5 --hidden_dim 100 --nepoch 50 --model_type lenet --model_name LeNetC --lr 0.001 --batch_size 32 --run_seed 4 --rand_split --agent_name OGD --agent_type ogd_plus --no_val --wandb_dryrun --ogd --subset_size 1