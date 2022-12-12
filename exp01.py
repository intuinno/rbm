import time
import os
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timeit
import numpy as np
from utils import tile_raster_images
from torch.utils.tensorboard import SummaryWriter 

from models.rbm import RBM 

localtime = time.asctime(time.localtime(time.time()))
Seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name",
                    help="name of experiment. ex) exp-01-PCD10",
                    default="exp")
parser.add_argument("-e", "--epoch", 
                    type=int,
                    help="Number of the epoch. default=15",
                    default=15)
parser.add_argument("-b", "--batch",
                    type=int,
                    help="mini batch size. default=20",
                    default="20")
parser.add_argument("-l", "--learning_rate",
                    help="Learning rate.  default=0.1",
                    type=float,
                    default=0.1)
parser.add_argument("-v", "--num_vis",
                    type=int,
                    help="Number of visual variables. default=784",
                    default=784)
parser.add_argument("-s", "--num_hidden",
                    type=int,
                    help="Number of the hidden (latent) variables. default=500",
                    default=500)
parser.add_argument("-k", "--num_gibbs",
                    type=int,
                    help="Number of the Gibbs sampling per iteration.  default=15",
                    default=15)
parser.add_argument("--use_PCD", dest='USE_PCD', action='store_true')
parser.add_argument("--use_CD", dest='USE_PCD', action='store_false')
parser.set_defaults(USE_PCD=True)


args = parser.parse_args()
num_epoch = args.epoch
batch_size = args.batch
expName = args.name 
n_vis = args.num_vis
n_hid = args.num_hidden
k = args.num_gibbs
lr = args.learning_rate

expDir = 'save/' + expName + '_' + localtime
writer = SummaryWriter('../rbm_runs/' + expDir, comment=expName)

# os.makedirs('./save', exist_ok=True)
# os.makedirs('./save/' + expDir)
archiveFilePath = './save/' + expDir + '.pt'



def train(dataloader, model, optimizer, persistent):
    size = len(dataloader.dataset)
    
    reconstruction_cost = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        v, model_v, pre_v_h, persistent = model(X, persistent)
        model.train()
        fv = model.calc_free_energy(v)
        # chain_end = model_v.detach()
        model_fv = model.calc_free_energy(model_v)
        loss = torch.mean(fv - model_fv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     reconstruction_cost += get_reconstruction_cost(v, pre_v_h)
        model.eval()
        reconstruction_cost += get_reconstruction_cost(v, pre_v_h) 
    return reconstruction_cost/size, persistent
        
def get_reconstruction_cost(target, pred):
    cross_entropy = F.binary_cross_entropy(pred, target, reduction='sum')
    return cross_entropy



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
    plt.show()
    
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
    filter = model.W.reshape(n_hid, 1, 28,28)
    filter_tile = make_grid(filter,
                        nrow=20,
                        normalize=True,
                        scale_each=True)  
    filename = f'filter_at_epoch_{epoch}.png'
    save_tensor(filter_tile, filename)
    
def save_filter_with_raster(epoch):
    filter_tile =  tile_raster_images(
                X=model.W.detach().numpy(),
                img_shape=(28, 28),
                tile_shape=(25, 20),
                tile_spacing=(1, 1)
            )
    filename = f'filter_at_epoch_{epoch}.png'
    plt.imsave(filename, filter_tile, cmap='gray')
    
    
def sample_rbm(model, test_loader, epoch, num_sample_row=20):
    with torch.no_grad():
        sample_iteration = 2
        persistent_chain = torch.zeros(batch_size, n_hid)
        v = next(iter(test_loader))[0]

        original = make_grid(v.view(batch_size, 1, 28,28).data, nrow=batch_size)
        
        for i in range(num_sample_row):
            v, model_v, p_v_h = model.sample(v,sample_iteration)      
            sample = make_grid(p_v_h.view(batch_size, 1, 28,28).data, nrow=batch_size)
            original = torch.cat((original, sample), 1)

        save_tensor(original, f'sample_at_{epoch}.png')
        
 

training_data = datasets.MNIST(
                    root="data",
                    train = True,
                    download = True, 
                    transform = transforms.ToTensor()
                    )

train_dataloader = DataLoader(training_data, batch_size=batch_size, drop_last=True, shuffle=True)

test_data = datasets.MNIST(
                    root="data",
                    train = False,
                    download = True, 
                    transform = transforms.ToTensor()
                    )


test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

start_time = timeit.default_timer()

if not os.path.isdir(expDir):
    os.makedirs(expDir)
os.chdir(expDir)


model = RBM(n_vis=n_vis, n_hid=n_hid, kCD=k).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
persistent_chain = torch.zeros(batch_size, n_hid)
save_filter(-1)
print("Starting training")
sample_rbm(model, test_loader, -1)
for i in range(num_epoch):
    cost, persistent_chain = train(train_dataloader, model, optimizer, persistent_chain)
    current_time = timeit.default_timer()
    print(f"Done {i}th epoch: Reconstruction Loss: {cost} Time: {current_time - start_time}")
    save_filter_with_raster(i)    
    sample_rbm(model, test_loader, i)
    
    

os.chdir('../')



