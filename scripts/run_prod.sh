# Permuted MNIST

n_permutation=15

 python main.py --n_permutation $n_permutation --force_out_dim 10 --no_class_remap \
 --group_id "permute-mnist-sgd" --dataset "MNIST" --run_name "permute-mnist-sgd"

  python main.py --n_permutation $n_permutation --force_out_dim 10 --no_class_remap --ogd \
 --group_id "permute-mnist-ogd" --dataset "MNIST" --run_name "permute-mnist-ogd" --ogd

   python main.py --n_permutation $n_permutation --force_out_dim 10 --no_class_remap --ogd_plus \
 --group_id "permute-mnist-ogd+" --dataset "MNIST" --run_name "permute-mnist-ogd+"



# SPLIT CIFAR

#