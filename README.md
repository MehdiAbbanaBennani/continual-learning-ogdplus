# Continual Learning with OGD and OGD+

This is the official implementation of the article [Generalisation Guarantees for Continual Learning with Orthogonal
 Gradient
 Descent](https://arxiv.org/abs/2006.11942) in PyTorch.
 
## Requirements
- PyTorch >= 1.5.0 
- [Typed Argument Parser](https://github.com/swansonk14/typed-argument-parser)
- wandb


## Reproducibility
In order to replicate the results of the paper, please refer to the scripts provided in the scripts
 directory.
- The production scripts have a prefix *prod*.
- The ablation studies scripts have a prefix *ablation*.
- The grid search scripts have a prefix *gs*.
 
## Questions/ Bugs
- For questions or bugs, please feel free to contact me or to raise an issue on Github :)


## Licence
### Continual-Learning-Benchmark
A substantial part of this source code was initially forked from the repository [GT-RIPL/Continual-Learning-Benchmark
](https://github.com/GT-RIPL/Continual-Learning-Benchmark). The corresponding Licence is also
 provided in the root directory.  
 The work related to the original source code is the following : 
 ```
@inproceedings{Hsu18_EvalCL,
  title={Re-evaluating Continual Learning Scenarios: A Categorization and Case for Strong Baselines},
  author={Yen-Chang Hsu and Yen-Cheng Liu and Anita Ramasamy and Zsolt Kira},
  booktitle={NeurIPS Continual learning Workshop },
  year={2018},
  url={https://arxiv.org/abs/1810.12488}
}
```
 
It was released under The MIT License found in the LICENSE file in the root directory of this
 source tree. 
 
 ### stable-continual-learning
 The [Stable SGD](https://arxiv.org/abs/2006.06958) code in the *external* folder was forked from
  this [repository](https://github.com/imirzadeh/stable-continual-learning/tree/master/stable_sgd).
 
 Since I brought some changes to it, for reproducibility experiments of the original [paper](https://arxiv.org/abs/2006.06958), I
  recommend to fork the original codebase. This fork may also not be up to date.
 
 **The modifications I brought were for
 logging, consistency with the other benchmarks and in order
 to run the experiments on other datasets.** 
 
  Please let me know if you have any issues :)
  
  
## Citation
If this repository helps your work, please cite:

```
@misc{bennani2020generalisation,
      title={Generalisation Guarantees for Continual Learning with Orthogonal Gradient Descent}, 
      author={Mehdi Abbana Bennani and Thang Doan and Masashi Sugiyama},
      year={2020},
      eprint={2006.11942},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```