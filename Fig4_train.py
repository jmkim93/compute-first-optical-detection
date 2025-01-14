#%% Dataset
import numpy as np
from numpy.random import rand
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optics_module as om
import torchvision.datasets as ds


torch.set_default_device("cuda:3")
device = torch.device("cuda:3")

import optics_module as om

import torchvision.datasets as ds


emnist_train = ds.EMNIST(root="/home/jungminkim/Documents/NoisyIR/emnist", 
                         split='digits', download=True, train=True)
emnist_test = ds.EMNIST(root="/home/jungminkim/Documents/NoisyIR/emnist", 
                        split='digits', download=True, train=False)
N_class = emnist_train.targets.max().item()+1
train_set = emnist_train.data[torch.argsort(emnist_train.targets)]+0.0
train_set = train_set.reshape(N_class, -1, 28, 28)
train_set, val_set = train_set[:,:5000], train_set[:,5000:6000]

test_set = emnist_test.data[torch.argsort(emnist_test.targets)]+0.0
test_set = test_set.reshape(N_class, -1, 28, 28)[:,:1000]
del emnist_train, emnist_test


#### Low contrast and unnormalized input
Tmin = 300
Tmax = 310
ambient = om.photon_num_per_vol(10, torch.tensor(Tmin)).to('cpu')
train_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*train_set/255)/ambient)**0.5
val_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*val_set/255)/ambient)**0.5
test_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient)**0.5
torch.cuda.empty_cache()
I_diff = (train_set.abs().max()**2-train_set.abs().min()**2).item()




#%% Train

I_sat = 0.5
p = 0.2
N_modes_optics = [784, 784]
N_modes_digital = [784, 300, 200, 50, 10]

type_init = "R"
dt = 1e4
# dark_noise_level = np.arange(0, 0.21, 0.04)
dark_noise_level = np.array([0.01, 0.1, 0.2])

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
N_epoch = 2000
lr = 4e-6
batch_size = 5000
train_optical = True

##### Dataset #####
X = train_set.reshape(-1,28,28).reshape(-1,28*28)
Y = ((torch.arange(N_class).reshape(-1,1) * torch.ones(1,train_set.shape[1])).reshape(-1,1) == torch.arange(N_class).reshape(1,-1) ) + 0.0
X_val = val_set.reshape(-1,28,28).reshape(-1,28*28)
Y_val = ((torch.arange(N_class).reshape(-1,1) * torch.ones(1,val_set.shape[1])).reshape(-1,1) == torch.arange(N_class).reshape(1,-1) )+0.0

dataset = TensorDataset(X, Y)
dataset_val = TensorDataset(X_val, Y_val)
X_test = test_set.reshape(-1,28,28).reshape(-1,28*28)
Y_test = (torch.arange(N_class).reshape(-1,1) * torch.ones(1,test_set.shape[1])).reshape(-1).cpu()

loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
loader_val = DataLoader(dataset_val, batch_size=10000, shuffle=False, generator=torch.Generator(device=device))


for ens in [5]:
    for jj, I_dark in enumerate(dark_noise_level):
        print(ens, I_dark)
        net = om.classifier(
            N_modes_optics, N_modes_digital, I_sat, dt, I_dark, p, 
            optical=True, digital=True, batchnormfirst=True, affine=False
        )
        
        net[0][0].parametrizations.weight[0].base = om.HaarRandU(seed=ens)
        net[0][0].parametrizations.weight[0].requires_grad = train_optical

        optimizer = torch.optim.Adam((net if train_optical else net[-1]).parameters(), lr=lr)
        train_loss, val_loss, opt_state_dict = om.run(net, optimizer, criterion, N_epoch, loader, loader_val, device, real=False)
        N_add = 0
        while np.argmin(val_loss) > max(len(val_loss)-100, len(val_loss)*0.85) and N_add<15:
            N_add += 1
            N_epoch_add = 200
            train_loss, val_loss, opt_state_dict = om.run(
                net, optimizer, criterion, N_epoch_add, loader, loader_val, device, real=False,
                resume=True, trainloss=train_loss, valloss=val_loss, optstatedict=opt_state_dict
            )
        torch.cuda.empty_cache()

        torch.save({
            "epoch": N_epoch,
            "net_state_dict": net.state_dict(),
            "net_state_dict_opt": opt_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        }, "Data_Fig4/net_random_darknoise{}_ens{}.pt".format(I_dark, ens))


        net.load_state_dict(opt_state_dict)
        net.eval()
        with torch.no_grad():
            y = net(X_test.to(device)).cpu()
            test_infer = torch.argmax(y, dim=1)
        test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
        print("Test accuracy: {}%".format( round(test_accuracy*100, 4) ))
