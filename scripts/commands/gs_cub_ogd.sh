group_tag=$(date +'%Y-%m-%d-%H-%M')


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 10 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 64 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 256 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 10 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 64 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 256 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 10 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 64 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.sgd.$group_tag --run_name gs_split-cub.sgd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 256 --run_seed 0    --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 10 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 64 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 256 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 10 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 64 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 256 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 10 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 64 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd.$group_tag --run_name gs_split-cub.ogd.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 256 --run_seed 0   --ogd --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 10 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 64 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.001  --reg_coef 0 --batch_size 256 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 10 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 64 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.01  --reg_coef 0 --batch_size 256 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 10 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 64 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun


python main.py --memory_size 100 --group_id gs_split-cub.ogd-plus.$group_tag --run_name gs_split-cub.ogd-plus.$group_tag --is_split --force_out_dim 0 --first_split_size 20 --other_split_size 20 --hidden_dim 256 --model_type alexnet  --nepoch 75 --model_name AlexNetCL --lr 0.1  --reg_coef 0 --batch_size 256 --run_seed 0   --ogd_plus --agent_name OGD --agent_type ogd_plus --no_val --is_split_cub --wandb_dryrun

