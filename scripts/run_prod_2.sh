# SPLIT MNIST

memory_size=200

python main.py --memory_size $memory_size --group "split-mnist-ogd+"  --dataset "MNIST" --ogd_plus \
  --run_name "split-mnist-ogd+" --is_split --force_out_dim 0 --dataset "MNIST" --first_split_size 2 \
  --other_split_size 2 --rand_split

python main.py --memory_size $memory_size --group "split-mnist-ogd"  --dataset "MNIST" --ogd \
--run_name "split-mnist-ogd" --is_split --force_out_dim 0 --dataset "MNIST" --first_split_size 2 \
--other_split_size 2 --rand_split

python main.py --memory_size $memory_size --group "split-mnist-sgd"  --dataset "MNIST" \
--run_name "split-mnist-sgd" --is_split --force_out_dim 0 --dataset "MNIST" --first_split_size 2 \
--other_split_size 2 --rand_split



# Rotated MNIST

n_rotate=15
rotate_step=5

 python main.py --n_rotate $n_rotate --rotate_step $rotate_step --memory_size 1000 --force_out_dim 10 \
 --no_class_remap --group_id "rotated-mnist-sgd" --dataset "MNIST" --run_name "rotated-mnist-sgd"

  python main.py --n_rotate $n_rotate --rotate_step $rotate_step --memory_size 1000 --force_out_dim 10 \
 --no_class_remap --group_id "rotated-mnist-ogd" --dataset "MNIST" --run_name "rotated-mnist-ogd" --ogd

 python main.py --n_rotate $n_rotate --rotate_step $rotate_step --memory_size 1000 --force_out_dim 10 \
 --no_class_remap --group_id "rotated-mnist-ogd+" --dataset "MNIST" --run_name "rotated-mnist-ogd+" --ogd_plus


