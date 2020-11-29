group_tag=$(date +'%Y-%m-%d-%H-%M')


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.0,1.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.0,1.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 0.1


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.1.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.1.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.10.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.10.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.100.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.100.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.1000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.1000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.10000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.10000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.100000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.100000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_EWC.1000000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_EWC.1000000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name EWC --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1000000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.0,1.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.0,1.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 0.1


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.1.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.1.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.10.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.10.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.100.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.100.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.1000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.1000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.10000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.10000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.100000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.100000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_MAS.1000000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_MAS.1000000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name MAS --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1000000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.0,1.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.0,1.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 0.1


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.1.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.1.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.10.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.10.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.100.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.100.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.1000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.1000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.10000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.10000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 10000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.100000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.100000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 100000


python main.py --force_out_dim 10 --no_class_remap --group_id gs_rotated-mnist_SI.1000000.$group_tag --dataset MNIST --n_rotate 15 --rotate_step 5 --run_name gs_rotated-mnist_SI.1000000.$group_tag --run_seed 0 --memory_size 100 --lr 0.001 --batch_size 32 --val_check_interval 10000000 --hidden_dim 100 --nepoch 5   --agent_name SI --agent_type regularization  --no_val --wandb_dryrun --reg_coef 1000000

