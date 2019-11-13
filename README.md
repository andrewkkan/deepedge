# Deep Edge Project 

This repo started from a github (https://github.com/shaoxiongji/federated-learning), but plenty of changes have been made.
The origianl repo from the above site could not reproduce any of the results from the Google paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) they claimed to be reproducing.
The accuracy was way off and they havent made much attempt to improve the issue.  They were mainly trying to focus on their own publication, which I hope they have a better code base for generating their results.

In any case, after a few corrections in their code, at least the MNIST results are looking better.  For example, the following runs will more or less reproduce the results from the Google papers for MNIST.

### IID MNIST E = 1 B = 10 C = 0.1
python main_fed.py --dataset mnist --model mlp --epochs 100 --gpu 0 --num_channels 3 --iid --local_bs 10 --frac 0.1 --local_ep 1 --momentum 0.5

### Non-IID MNIST E = 1 B = 10 C = 0.1
python main_fed.py --dataset mnist --model mlp --epochs 680 --gpu 0 --num_channels 3 --local_bs 10 --frac 0.1 --local_ep 1 --momentum 0.5


## References
```
@article{mcmahan2016communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, H Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and others},
  journal={arXiv preprint arXiv:1602.05629},
  year={2016}
}
```

## Requirements
python 3.6  
pytorch>=0.4
