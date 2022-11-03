#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import timeit
import numpy as np


# In[2]:


batch_size = 64
n_epochs = 10 
lr = 0.01
n_hid = 128
n_vis = 784
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[3]:



class RBM(nn.Module):
    
    def __init__(self, n_vis=784, n_hid=500, kCD=1):
        super(RBM, self).__init__()
        self.vbias = nn.Parameter(torch.randn(n_vis))
        self.hbias = nn.Parameter(torch.randn(n_hid))
        self.W = nn.Parameter(torch.randn(n_hid, n_vis))
        self.kCD = kCD 
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        
    def forward(self, v):
        v_flat = self.flatten(v)
        v_h = v_flat
        for _ in range(self.kCD):
            h_v, pre_h_v = self.sample_h_given_v(v_h)
            v_h, pre_v_h = self.sample_v_given_h(h_v)
        return v_flat, v_h, pre_v_h
    
    def sample_h_given_v(self, v):
        p = self.hbias  + torch.matmul(v,self.W.t())
        p = self.sigmoid(p)
        h = torch.bernoulli(p)
        return h, p
    
    def sample_v_given_h(self, h):
        p = self.vbias + h.mm(self.W)
        p = self.sigmoid(p)
        v = torch.bernoulli(p)
        return v, p
    
    def calc_free_energy(self, v):
        a = -torch.matmul(v,self.vbias)
        e = self.hbias + v.mm(self.W.t())
#         f = torch.log(1+torch.exp(e))
        f = self.softplus(e)
        g = torch.sum(f, dim=1)
        return a - g
    

        


# In[4]:


model = RBM(n_vis=n_vis, n_hid=n_hid, kCD=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[5]:


def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    
    reconstruction_cost = 0 
    
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        v, model_v, pre_v_h = model(X)
        model.train()
        fv = model.calc_free_energy(v)
        model_fv = model.calc_free_energy(model_v)
        loss = torch.mean(fv - model_fv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        reconstruction_cost += get_reconstruction_cost(v, pre_v_h)
        
    
    return reconstruction_cost.mean()
        
def get_reconstruction_cost(target, pred):
    cross_entropy = F.binary_cross_entropy(pred, target, reduction='sum')
    return cross_entropy
    
        
        
        
        


# \begin{align}
#                 F(x) &= -\log \sum_h \exp (-E(x, h)) \\
#                 &= -a^\top x - \sum_j \log (1 + \exp(W^{\top}_jx + b_j))\,.
#             \end{align}

# In[6]:


training_data = datasets.MNIST(
                    root="data",
                    train = True,
                    download = True, 
                    transform = transforms.ToTensor()
                    )

test_data = datasets.MNIST(
                    root="data",
                    train = False,
                    download = True, 
                    transform = transforms.ToTensor()
                    )



batch_size = 100
train_dataloader = DataLoader(training_data, batch_size=batch_size)




num_epoch = 10
plotting_time = 0
start_time = timeit.default_timer()
for i in range(num_epoch):
    print("Starting training")
    cost = train(train_dataloader, model, optimizer)
    
    current_time = timeit.default_timer()
    print(f"Done {i}th epoch: Reconstruction Loss: {cost} Time: {current_time - start_time}")


def show_and_save(img, file_name):
    r"""Show and save the image.
    Args:
        img (Tensor): The image.
        file_name (Str): The destination.
    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg, cmap='gray')
    plt.imsave(f, npimg)




