import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

from IPython import embed


# Creating the artificial dataset
class DataLinRegress(Dataset):
    
    # Constructor
    def __init__(self, num_inputs, noise_sigma=0.1, num_data=1000, num_targets=10):
        """
        For a linear regressio dataset, the inputs are the x's and the outputs are the y's.  Simlar to object recognition, the x's are the images,
        and the y's are the targets or labels.

        However, we use num_targets for a different purpose.  So please heed the possible confusion.  Here,
        num_targets has no meaning to the dataset, since there is no actual target or label in linear regression like you would for object recognition.
        However, the concept of labels is used for creating non-IID datasets.  See function generic_noniid in utils/sampling.py for implementation.
        Same concept with self.classes
        """
        bound = 256.0
        self.x = torch.rand(num_data,num_inputs) * 2.0 * bound - bound
        
        self.w = torch.ones(num_inputs, 1)
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b

        self.y = self.f + noise_sigma*torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]

        self.targets = KMeans(n_clusters=num_targets).fit_predict(self.x)
        self.classes = set(self.targets)
    
    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # Getting the length
    def __len__(self):
        return self.len



# Defining a function for plotting the plane
def plane2D(model,dataset,n=0):
      
    w1 = model.state_dict()['linear.weight'].numpy()[0][0]
    w2 = model.state_dict()['linear.weight'].numpy()[0][1]
    b = model.state_dict()['linear.bias'].numpy()
    
    x1 = dataset.x[:,0].view(-1,1).numpy()
    x2 = dataset.x[:,1].view(-1,1).numpy()
    y = dataset.y.numpy()
    X,Y = np.meshgrid(np.arange(x1.min(),x1.max(),0.05),np.arange(x2.min(), x2.max(), 0.05))
    
    yhat = w1*X + w2*Y + b
    
    # plotting
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    ax.plot(x1[:,0], x2[:,0], y[:,0], 'ro', label = 'y')
    ax.plot_surface(X,Y,yhat)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.title('Estimated plane iteration: '+ str(n))
    plt.show()


# Creating a linear regression model
class lin_reg(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(lin_reg, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        
    def forward(self,x):
        yhat = self.linear(x)
        return yhat




def my_criterion(yhat,y):
    out = torch.mean((yhat - y)**2)
    return out

def train_model(epochs):
    for epoch in range(epochs):
        for x,y in trainloader:
            yhat = model(x)
            loss = criterion(yhat,y)
            Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":

    # Instantiation of the class  
    my_data = DataLinRegress(20)
    # Instantiation of an object
    model = lin_reg(2,1)
    print("The parameters: ", model.state_dict())
    # Parameters
    criterion = nn.MSELoss()
    criterion = my_criterion
    optimizer = optim.SGD(model.parameters(), lr = 0.01) 
    # Training data object which loads the artificial data
    trainloader = DataLoader(dataset = my_data, batch_size = 2)
    # Training the model
    Loss = []  # variable for storing losses after each epoch
    epochs = 100
    print('Before training:')
    plane2D(model, my_data)

    # Calling the training function          
    train_model(epochs)
    print("After training: ")
    plane2D(model, my_data, epochs)




        