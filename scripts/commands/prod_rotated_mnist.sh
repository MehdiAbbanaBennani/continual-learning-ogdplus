group_tag=$(date +'%Y-%m-%d-%H-%M')


python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.sgd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.sgd.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5  --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.sgd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.sgd.$group_tag --run_seed 1 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5  --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.sgd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.sgd.$group_tag --run_seed 2 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5  --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.sgd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.sgd.$group_tag --run_seed 3 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5  --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.sgd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.sgd.$group_tag --run_seed 4 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5  --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0


python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd.$group_tag --run_seed 1 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd.$group_tag --run_seed 2 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd.$group_tag --run_seed 3 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd.$group_tag --run_seed 4 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0


python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd-plus.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd-plus.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd_plus --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd-plus.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd-plus.$group_tag --run_seed 1 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd_plus --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd-plus.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd-plus.$group_tag --run_seed 2 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd_plus --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd-plus.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd-plus.$group_tag --run_seed 3 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd_plus --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ogd-plus.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ogd-plus.$group_tag --run_seed 4 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5 --ogd_plus --agent_name OGD --agent_type ogd_plus  --no_val --wandb_dryrun --reg_coef 0.0


python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ewc.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ewc.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ewc.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ewc.$group_tag --run_seed 1 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ewc.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ewc.$group_tag --run_seed 2 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ewc.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ewc.$group_tag --run_seed 3 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.ewc.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.ewc.$group_tag --run_seed 4 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100


python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.si.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.si.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.si.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.si.$group_tag --run_seed 1 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.si.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.si.$group_tag --run_seed 2 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.si.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.si.$group_tag --run_seed 3 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.si.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.si.$group_tag --run_seed 4 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10


python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.mas.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.mas.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.mas.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.mas.$group_tag --run_seed 1 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.mas.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.mas.$group_tag --run_seed 2 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.mas.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.mas.$group_tag --run_seed 3 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
python main.py --force_out_dim 10 --no_class_remap --group_id rotated-mnist.mas.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name rotated-mnist.mas.$group_tag --run_seed 4 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10
