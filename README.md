# Deep Edge Project 

This repo started from a github (https://github.com/shaoxiongji/federated-learning), but plenty of changes have been made.
The origianl repo from the above site could not reproduce any of the results from the Google paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) they claimed to be reproducing.
The accuracy was way off and they havent made much attempt to improve the issue.  They were mainly trying to focus on their own publication, which I hope they have a better code base for generating their results.

In any case, after a few corrections in their code, at least the MNIST results are looking better.  For example, the following runs will more or less reproduce the results from the Google papers for MNIST.

### IID MNIST E = 1 B = 10 C = 0.1
python main_fed.py --dataset mnist --model mlp --epochs 100 --gpu 0 --num_channels 1 --iid --local_bs 10 --frac 0.1 --local_ep 1 --momentum 0.5

### Non-IID MNIST E = 1 B = 10 C = 0.1
python main_fed.py --dataset mnist --model mlp --epochs 700 --gpu 0 --num_channels 1 --local_bs 10 --frac 0.1 --local_ep 1 --momentum 0.5

## Try out these new flags!!

### --async_s2d 0 or 1 or 2
This flags turns on asynchronous server-to-device parameter updates. 0 is more benign, 1 is more hostile, 2 is most hostile.
- 0: Synchronous mode.  The order of operations are as followed: (1) Weight update from server to selected devices; (2) Selected devices training; (3) FedAvg on selected devices
- 1: Asynchronous mode 1. The order of operations are as followed: (1) Selected devices training; (2) FedAvg on selected devices (3) Weight update from server to selected devices
- 2: Asynchronous mode 2. The order of operations are as followed:  (1) Selected devices training; (2) Weight update from server to selected devices; (3) FedAvg on selected devices
### --sync_params
This flag tests out a mode in which weight update happens after every batch of training.  Basically same order of operations as synchronous mode 0, but the operations happen after every batch.
### --rand_d2s
This flags turns on random device participation as you would expect in an asynchronous environment.  The random arrival at the server is binomial.  This is modelled by success and failure probability at the local devices.  Each local device has success update prob = C, and failure update prob = 1-C.  With "--rand_d2s" alone without any value provided, all local devices will have the same success prob = C.  With "--rand_d2s 0.1 0.2 0.3", for example, local device #1 will get success prob = 0.1, #2 will be 0.2, #3 will be 0.3, #4 will be 0.1, etc, all the way through 100 for num_users = 100. Each communication still has a minimal device of 1.  FedAvg would just average whatever devices it gets in that round.
### --fedmas 1.0
This flag implements the Memory Aware Synapses algorithm for catastrophic forgetting.  arXiv:1812.03596

## References
```
@article{mcmahan2016communication,
  title={Communication-efficient learning of deep networks from decentralized data},
  author={McMahan, H Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and others},
  journal={arXiv preprint arXiv:1602.05629},
  year={2016}
}
@ARTICLE{2018arXiv181203596A,
       author = {{Aljundi}, Rahaf and {Kelchtermans}, Klaas and {Tuytelaars}, Tinne},
        title = "{Task-Free Continual Learning}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence, Computer Science - Machine Learning, Statistics - Machine Learning},
         year = "2018",
        month = "Dec",
          eid = {arXiv:1812.03596},
        pages = {arXiv:1812.03596},
archivePrefix = {arXiv},
       eprint = {1812.03596},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018arXiv181203596A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Requirements
python 3.6  
pytorch>=0.4
