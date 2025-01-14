#%% Dataset

import numpy as np
from numpy.random import rand
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import matplotlib
# import plot_setting
# import scipy

# from tqdm import tqdm

import torch
# from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_device("cuda:3")
device = torch.device("cuda:3")

import optics_module as om

import torchvision.datasets as ds
# from  torch.nn.modules.upsampling import Upsample

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

### Low contrast and unnormalized input - Fig2
Tmin = 300
Tmax = 310
ambient = om.photon_num_per_vol(10, torch.tensor(Tmin)).to('cpu')
train_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*train_set/255)/ambient)**0.5
val_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*val_set/255)/ambient)**0.5
test_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient)**0.5
torch.cuda.empty_cache()
I_diff = (train_set.abs().max()**2-train_set.abs().min()**2).item()

### amplitude modulated signal input - Supp Fig
# train_set = train_set/255 + 0j
# val_set = val_set/255 + 0j
# test_set = test_set/255 + 0j
# I_diff = (train_set.abs().max()**2-train_set.abs().min()**2).item()

### phase modulated signal input
# train_set = 2*(train_set>=128)  -1 + 0j
# val_set = 2*(val_set>=128)-1 + 0j
# test_set = 2*(test_set>=128)-1 + 0j


#### 2D FFT and Identity matrix
N = 28
Id = torch.eye(N**2)+0j
Id_real = torch.real(Id)

F, det_filt = {}, {}
F["1"], det_filt["1"] = om.Fourier_block([28])
# F["5"], det_filt["5"] = om.Fourier_block([5, 6, 6, 6, 5])
# F["6"], det_filt["6"] = om.Fourier_block([4,5,5,5,5,4])
F["7"], det_filt["7"] = om.Fourier_block([4]*7)
# F["8"], det_filt["8"] = om.Fourier_block([3,3,4,4,4,4,3,3])
# F["9"], det_filt["9"] = om.Fourier_block([3,3,3,3,4,3,3,3,3])
F["10"], det_filt["10"] = om.Fourier_block([2,3,3,3,3,3,3,3,3,2])
# F["11"], det_filt["11"] = om.Fourier_block([3,3,3,2,2,2,2,2,3,3,3])
# F["12"], det_filt["12"] = om.Fourier_block([2,2,2,2,3,3,3,3,2,2,2,2])
# F["13"], det_filt["13"] = om.Fourier_block([3,2,2,2,2,2,2,2,2,2,2,2,3])
# F["14"], det_filt["14"] = om.Fourier_block([2]*14)


#%% Fig 2 Network training and model generation

train_optical_list = [True, True, False, False, False][0:1]
type_init_list = ["F1", "R", "I", "F10", "F7"][0:1]

for type_init, train_optical in zip(type_init_list, train_optical_list):

    N_filt = int(type_init[1:]) if (not train_optical and type_init[0]=="F") else 28
    filt = det_filt[type_init[1:]] if (not train_optical and type_init[0]=="F") else np.arange(784)
    Unitary_init = F[type_init[1:]]*(type_init[0]=="F") if type_init[0]=="F" else Id*(type_init[0]=="I") + om.HaarRandU(seed=0)*(type_init=="R")

    I_sat = 0.5
    p = 0.2
    N_epoch = 15000 if train_optical else 2000
    N_modes_optics = [784]*2
    N_modes_digital = [N_filt**2, 300, 200, 50, 10]
    # N_modes_digital = [N_filt**2, 100, 40, 10] # Shallow
    # N_modes_digital = [N_filt**2, 300, 250, 200, 50, 10] # deeper
    batch_size = 5000

    print("Type:{}, Train optical: {}, ".format(type_init, train_optical))

    I_dark = I_diff/np.sqrt(2)
    dt = 1/(I_diff/np.sqrt(2))**2
    Noise_tot = np.sqrt(1/dt + I_dark**2)
    SNR_expected = 1/Noise_tot

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    lr = 2e-5 if train_optical else 4e-5

    net = om.classifier(
        N_modes_optics, N_modes_digital, I_sat, dt, I_dark, p, 
        optical=True, digital=True, batchnormfirst=True, affine=False, filter=filt
    )

    net[0][0].parametrizations.weight[0].base = Unitary_init
    net[0][0].parametrizations.weight[0].requires_grad = train_optical


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
    optimizer = torch.optim.Adam((net if train_optical else net[-1]).parameters(), lr=lr)

    ##### Training #####
    train_loss, val_loss, opt_state_dict = om.run(
        net, optimizer, criterion, N_epoch, loader, loader_val, device, 
        real=False, incoherent=True
    )
    N_add = 0
    while np.argmin(val_loss) > len(val_loss)*0.9 and N_add<30:
        N_add += 1
        N_epoch_add = 500
        train_loss, val_loss, opt_state_dict = om.run(
            net, optimizer, criterion, N_epoch_add, loader, loader_val, device, real=False,
            resume=True, incoherent=True,
            trainloss=train_loss, valloss=val_loss, optstatedict=opt_state_dict
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
    }, "Data_SuppFig_incoherence/net_type{}_train{}.pt".format(type_init,train_optical))


    ##### Evaluation #####
    net.load_state_dict(opt_state_dict)
    net.eval()
    with torch.no_grad():
        E_in = X_test.to(device)
        I_in = E_in.abs()**2
        W = (net[0][0].weight.abs()**2).T
        I_out = I_in @ W 
        I_out_detach = I_out.detach()
        poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
        gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
        x = I_out + poisson + gaussian
        y = net[3](net[2](x))
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
    print("Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    net[1].set(dt=(0.01*I_diff)**(-2), dark=(0.01*I_diff))
    with torch.no_grad():
        E_in = X_test.to(device)
        I_in = E_in.abs()**2
        W = (net[0][0].weight.abs()**2).T
        I_out = I_in @ W 
        I_out_detach = I_out.detach()
        poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
        gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
        x = I_out + poisson + gaussian
        y = net[3](net[2](x))
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
    print("Zero-noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    net[1].set(dt=(I_diff)**(-2), dark=(0.01*I_diff))
    with torch.no_grad():
        E_in = X_test.to(device)
        I_in = E_in.abs()**2
        W = (net[0][0].weight.abs()**2).T
        I_out = I_in @ W 
        I_out_detach = I_out.detach()
        poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
        gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
        x = I_out + poisson + gaussian
        y = net[3](net[2](x))
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
    print("MAx Shot noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    net[1].set(dt=(0.01*I_diff)**(-2), dark=(I_diff))
    with torch.no_grad():
        E_in = X_test.to(device)
        I_in = E_in.abs()**2
        W = (net[0][0].weight.abs()**2).T
        I_out = I_in @ W 
        I_out_detach = I_out.detach()
        poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
        gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
        x = I_out + poisson + gaussian
        y = net[3](net[2](x))
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
    print("Max dark noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    torch.cuda.empty_cache()
