# Deep Edge Project 

This repo started from a github (https://github.com/shaoxiongji/federated-learning), but plenty of changes have been made.
The origianl repo from the above site could not reproduce any of the results from the Google paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) they claimed to be reproducing.
The accuracy was way off and they havent made much attempt to improve the issue.  They were mainly trying to focus on their own publication, which I hope they have a better code base for generating their results.

In any case, after a few corrections in their code, at least the MNIST results are looking better.  For example, the following runs will more or less reproduce the results from the Google papers for MNIST.

### IID MNIST E = 1 B = 10 C = 0.1
python main_fed.py --dataset mnist --model mlp --epochs 100 --gpu 0 --num_channels 3 --iid --local_bs 10 --frac 0.1 --local_ep 1 --momentum 0.9

### Non-IID MNIST E = 1 B = 10 C = 0.1
python main_fed.py --dataset mnist --model mlp --epochs 700 --gpu 0 --num_channels 3 --local_bs 10 --frac 0.1 --local_ep 1 --momentum 0.9

## Try out these new flags!!
###--async_s2d 
This flags turns on asynchronous server-to-device parameter updates.  Without this flag, all devices get global param sync from server in every communication round.  With this flag, only devices that update their local weight changes to the server get param sync from the server.  This means that some of these devices (called "stragglers" in literature) may have been working off of stale parameters, and their new local updates may be very localized.
###--rand_d2s 
This flags turns on random device participation as you would expect in an asynchronous environment.  The random arrival is modeled by a Poisson process, with lambda set to "C".  Let's say we have 100 users and C = 0.1, lambda would be 10 and you get an average of 10 devices on the average per communication round.  Each communication still has a minimal device of 1.  FedAvg would just average whatever devices it gets in that round.


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
