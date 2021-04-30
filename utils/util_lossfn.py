import torch
import torch.nn.functional as F

def CrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]            # batch_size
    outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
    outputs = outputs[range(batch_size), labels] # labels is 1-hot so the rest softmax outputs go to zero
    return -torch.sum(outputs)/float(batch_size) 

def MSELoss(yhat, y):
    batch_size = yhat.size()[0]
    return torch.sum((yhat - y).pow(2))/float(batch_size)

def MSESaddleLoss(yhat, y):
    batch_size = yhat.size()[0]
    return torch.sum((yhat[:,0] - y[:,0]).pow(2))/float(batch_size) - torch.sum((yhat[:,1] - y[:,1]).pow(2))/float(batch_size)
