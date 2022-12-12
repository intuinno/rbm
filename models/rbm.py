import torch
from torch import nn



class RBM(nn.Module):
    
    def __init__(self, n_vis=784, n_hid=500, kCD=1):
        super(RBM, self).__init__()
        self.vbias = nn.Parameter(torch.zeros(n_vis))
        self.hbias = nn.Parameter(torch.zeros(n_hid))

        self.W = 0.2 * nn.Parameter(torch.randn(n_hid, n_vis))

        self.kCD = kCD 
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        
    def forward(self, v, persistent=None):
        v_flat = self.flatten(v)
        v_h = v_flat
        
        if persistent is None:
            h_v, pre_h_v = self.sample_h_given_v(v_h)
            chain_start = h_v
        else:
            chain_start = persistent.detach()
            
        v_h, pre_v_h = self.sample_v_given_h(chain_start)
        
        for _ in range(self.kCD-1):
            h_v, pre_h_v = self.sample_h_given_v(v_h)
            v_h, pre_v_h = self.sample_v_given_h(h_v)
        return v_flat.detach(), v_h, pre_v_h, h_v.detach()
    
    def sample(self, v, k):
        v_flat = self.flatten(v)
        v_h = v_flat
        for _ in range(k):
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
    
