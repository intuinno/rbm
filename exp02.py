import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time 
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Hyper-parameters
num_epochs = 15
batch_size = 20
learning_rate = 0.1 
input_size = 784
hidden_size = 500
kCD = 15 
expName = 'RBMver2'

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)

# RBM class 
class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500, kCD=15):
        super(RBM, self).__init__()
        self.vbias = nn.Parameter(torch.zeros(n_vis))
        self.hbias = nn.Parameter(torch.zeros(n_hid))
        init_W = 0.2 * torch.randn(n_hid, n_vis)
        bounds = -4.0 * np.sqrt(6.0 / (n_vis + n_hid))
        self.W =  nn.Parameter(torch.FloatTensor(n_hid, n_vis).uniform_(bounds, -bounds))
        self.kCD = kCD
        
    def forward(self, v, persistent=None):
        v_h = v
        
        if persistent is None:
            h_v, pre_h_v = self.sample_h_given_v(v_h)
            chain_start = h_v
        else:
            chain_start = persistent
            
        v_h, pre_v_h = self.sample_v_given_h(chain_start)
        
        for _ in range(self.kCD-1):
            h_v, pre_h_v = self.sample_h_given_v(v_h)
            v_h, pre_v_h = self.sample_v_given_h(h_v)
        return v_h, pre_v_h, h_v
    
    def sample(self, v, k):
        v_h = v
        for _ in range(k):
            h_v, pre_h_v = self.sample_h_given_v(v_h)
            v_h, pre_v_h = self.sample_v_given_h(h_v)
        return  v_h, pre_v_h
    
    def sample_h_given_v(self, v):
        p = self.hbias  + torch.matmul(v,self.W.t())
        p = torch.sigmoid(p)
        h = torch.bernoulli(p)
        return h, p
    
    def sample_v_given_h(self, h):
        p = self.vbias + h.mm(self.W)
        p = torch.sigmoid(p)
        v = torch.bernoulli(p)
        return v, p
    
    def calc_free_energy(self, v):
        a = -torch.matmul(v,self.vbias)
        e = self.hbias + v.mm(self.W.t())
#         f = torch.log(1+torch.exp(e))
        f = F.softplus(e)
        g = torch.sum(f, dim=1)
        return a - g
    
    def get_reconstruction_cost(self, v):
        h_v, pre_h_v = self.sample_h_given_v(v)
        v_h, pre_v_h = self.sample_v_given_h(h_v)
        return F.binary_cross_entropy(pre_v_h, v, reduction="sum") 
        

def save_tensor(img, file_name):
    r"""Save the image.
    Args:
        img (Tensor): The image.
        file_name (Str): The destination.
    """
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imsave(f, npimg)   
    
def save_filter(epoch):
    filter = model.W.reshape(hidden_size, 1, 28,28)
    filter_tile = make_grid(filter,
                        nrow=20,
                        normalize=True,
                        scale_each=True)  
    filename = f'filter_at_epoch_{epoch}.png'
    save_tensor(filter_tile, filename)

def sample_rbm(model, test_loader, epoch, num_sample_row=20):
    with torch.no_grad():
        sample_iteration = 1000
        persistent_chain = torch.zeros(batch_size, hidden_size)
        v = next(iter(test_loader))[0]

        original = make_grid(v.view(batch_size, 1, 28,28).data, nrow=batch_size)
        v = v.view(-1, input_size)        
        for i in range(num_sample_row):
            model_v, p_v_h = model.sample(v,sample_iteration)      
            sample = make_grid(p_v_h.view(batch_size, 1, 28,28).data, nrow=batch_size)
            original = torch.cat((original, sample), 1)

        save_tensor(original, f'sample_at_{epoch}.png')
        

        

model = RBM(n_vis=input_size, n_hid=hidden_size, kCD=kCD).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
persistent = torch.zeros(batch_size, hidden_size, requires_grad=False)


localtime = time.asctime(time.localtime(time.time()))
expDir = 'save/' + expName + '_' + localtime

if not os.path.isdir(expDir):
    os.makedirs(expDir)
os.chdir(expDir)

save_filter(-1)
sample_rbm(model, test_loader, -1)

# Start training
print("Starting training")
train_size = len(train_loader.dataset)
for epoch in range(num_epochs):
    reconstruction_cost = 0
    for i, (x, _) in enumerate(train_loader):
        # Forward pass
        x = x.to(device).view(-1, input_size)
        model_v, pre_v_h, persistent = model(x, persistent.detach())
        fv = model.calc_free_energy(x)
        model_fv = model.calc_free_energy(model_v.detach())
        
        # Backprop and optimize
        loss = torch.mean(fv - model_fv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            reconstruction_cost += model.get_reconstruction_cost(x)
        
    with torch.no_grad():
        avg_cost = reconstruction_cost / train_size
        print ("Epoch[{}/{}], Reconst Loss: {:.4f}" 
                    .format(epoch+1, num_epochs, avg_cost))
        save_filter(epoch)
        sample_rbm(model, test_loader, epoch)
        
        