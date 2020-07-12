

 # Split CIFAR100
 # SGD
 NAME=debug-cifar100-sgd
 python main.py --val_size 2500 --repeat 1 --workers 16 --memory_size 1000 --hidden_dim \
 100 --agent_type ogd_plus --agent_name OGD --group_id $NAME \
 --dataset "CIFAR100" --run_name $NAME --is_split --force_out_dim 0 --dataset "CIFAR100" \
 --first_split_size 10 --other_split_size 10 --nepoch 50 --model_type "resnet" --model_name "ResNet20_cifar" \
 --rand_split --lr 0.01 --ogd_plus

 python main.py --memory_size $memory_size --group "split-mnist-sgd"  --dataset "MNIST" \
--run_name "split-mnist-sgd" --is_split --force_out_dim 0 --dataset "MNIST" --first_split_size 2 \
--other_split_size 2 --rand_split

memory_size=1000
name="split-cifar100-ogd-10ov2mx6"
subset_size=5
hidden_dim=200
nepoch=50
 python main.py --memory_size $memory_size --group $name  --dataset "CIFAR100" \
--run_name $name --is_split --force_out_dim 0 --first_split_size $subset_size \
--other_split_size $subset_size --rand_split --hidden_dim $hidden_dim --nepoch $nepoch \
--model_type "lenet" --model_name "LeNetC" --ogd \
--start_seed 5 --end_seed 7 --no_random_name

memory_size=1000
name="split-cifar100-ogd+-2lsjz6fp"
subset_size=5
hidden_dim=200
nepoch=50
 python main.py --memory_size $memory_size --group $name  --dataset "CIFAR100" \
--run_name $name --is_split --force_out_dim 0 --first_split_size $subset_size \
--other_split_size $subset_size --rand_split --hidden_dim $hidden_dim --nepoch $nepoch \
--model_type "lenet" --model_name "LeNetC" --ogd_plus \
--start_seed 5 --end_seed 7 --no_random_name