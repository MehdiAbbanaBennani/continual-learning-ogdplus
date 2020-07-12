memory_size=1000
subset_size=5
hidden_dim=200
nepoch=50

#name="split-cifar100-sgd-no-rand"
# python main.py --memory_size $memory_size --group $name  --dataset "CIFAR100" \
#--run_name $name --is_split --force_out_dim 0 --first_split_size $subset_size \
#--other_split_size $subset_size --hidden_dim $hidden_dim --nepoch $nepoch \
#--model_type "lenet" --model_name "LeNetC" \
#--start_seed 0 --end_seed 1

name="split-cifar100-ogd-no-rand"
 python main.py --memory_size $memory_size --group $name  --dataset "CIFAR100" \
--run_name $name --is_split --force_out_dim 0 --first_split_size $subset_size \
--other_split_size $subset_size --hidden_dim $hidden_dim --nepoch $nepoch \
--model_type "lenet" --model_name "LeNetC" --ogd \
--start_seed 0 --end_seed 1

name="split-cifar100-ogd+-no-rand"
 python main.py --memory_size $memory_size --group $name  --dataset "CIFAR100" \
--run_name $name --is_split --force_out_dim 0 --first_split_size $subset_size \
--other_split_size $subset_size --hidden_dim $hidden_dim --nepoch $nepoch \
--model_type "lenet" --model_name "LeNetC" --ogd_plus \
--start_seed 0 --end_seed 3

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

memory_size=1000
name="split-cifar100-sgd-zbejykxr"
subset_size=5
hidden_dim=200
nepoch=50
 python main.py --memory_size $memory_size --group $name  --dataset "CIFAR100" \
--run_name $name --is_split --force_out_dim 0 --first_split_size $subset_size \
--other_split_size $subset_size --rand_split --hidden_dim $hidden_dim --nepoch $nepoch \
--model_type "lenet" --model_name "LeNetC" \
--start_seed 5 --end_seed 7 --no_random_name


n_permutation=15
name="permuted-mnist-ogd+-lr0.1"
python main.py --n_permutation $n_permutation --force_out_dim 10 --no_class_remap --ogd_plus \
 --group $name --dataset "MNIST" --run_name $name --lr 0.1 --start_seed 0 --end_seed 1

 n_permutation=15
name="permuted-mnist-ogd+-lr0.001"
python main.py --n_permutation $n_permutation --force_out_dim 10 --no_class_remap --ogd_plus \
 --group $name --dataset "MNIST" --run_name $name --lr 0.001 --start_seed 0 --end_seed 1