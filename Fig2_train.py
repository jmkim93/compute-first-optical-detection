#%% Dataset

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import plot_setting
import scipy

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_device("cuda:0")
device = torch.device("cuda:0")

import optics_module as om

import torchvision.datasets as ds
from  torch.nn.modules.upsampling import Upsample

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

# fmnist_train = ds.FashionMNIST(root="/home/jungminkim/Documents/NoisyIR/fashionmnist", download=True, train=True)
# fmnist_test = ds.FashionMNIST(root="/home/jungminkim/Documents/NoisyIR/fashionmnist", download=True, train=False)
# N_class = fmnist_train.targets.max().item()+1
# train_set = fmnist_train.data[torch.argsort(fmnist_train.targets)]+0.0
# train_set = train_set.reshape(N_class, -1, 28, 28)
# train_set, val_set = train_set[:,:5000], train_set[:,5000:]

# test_set = fmnist_test.data[torch.argsort(fmnist_test.targets)]+0.0
# test_set = test_set.reshape(N_class, -1, 28, 28)
# del fmnist_train, fmnist_test


### amplitude modulated signal input - Supp Fig
# train_set = train_set / torch.sqrt((train_set**2).mean(dim=(2,3), keepdim=True)) + 0j
# val_set = val_set / torch.sqrt((val_set**2).mean(dim=(2,3), keepdim=True)) +0j
# test_set = test_set / torch.sqrt((test_set**2).mean(dim=(2,3), keepdim=True)) +0j

# train_set = train_set/255 + 0j
# val_set = val_set/255 + 0j
# test_set = test_set/255 + 0j
# I_diff = (train_set.abs().max()**2-train_set.abs().min()**2).item()

### phase modulated signal input
# train_set = 2*(train_set>=128)  -1 + 0j
# val_set = 2*(val_set>=128)-1 + 0j
# test_set = 2*(test_set>=128)-1 + 0j

## Low contrast and unnormalized input - Fig2
k_B = 8.617333262e-5 # [eV/K]
h = 4.135667696e-15 # [eV*s]
hbar = h/(2*np.pi)
c = 299792458e6 #[um/s] 
q_0 = 1.60217663e-19 # [coulombs]
occupation_num = lambda lda, T: 1/(torch.exp(h*c/(lda*k_B * T)) - 1)
std_num = lambda lda, T: torch.sqrt(occupation_num(lda,T)*(1+occupation_num(lda,T)))
DOS_per_vol = lambda lda: 8*np.pi/(lda**4)
photon_num_per_vol = lambda lda, T: occupation_num(lda, T)*DOS_per_vol(lda)
Tmin = 300
Tmax = 310
ambient = photon_num_per_vol(10, torch.tensor(Tmin)).to('cpu')
train_set = 0j+(photon_num_per_vol(10, Tmin+(Tmax-Tmin)*train_set/255)/ambient)**0.5
val_set = 0j+(photon_num_per_vol(10, Tmin+(Tmax-Tmin)*val_set/255)/ambient)**0.5
test_set = 0j+(photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient)**0.5
torch.cuda.empty_cache()
I_diff = (train_set.abs().max()**2-train_set.abs().min()**2).item()

#### 2D FFT and Identity matrix
N = 28
Id = torch.eye(N**2)+0j
Id_real = torch.real(Id)

def Fourier_block(n_list):
    N = sum(n_list)
    starting = np.concatenate([np.array([0]), np.cumsum(n_list)[0:-1]])
    ending = np.cumsum(n_list)
    k = torch.arange(N).reshape(-1,1,1,1)
    l = torch.arange(N).reshape(1,-1,1,1)
    m = torch.arange(N).reshape(1,1,-1,1)
    n = torch.arange(N).reshape(1,1,1,-1)
    part_x = [(k>=x0)*(m>=x0)*(k<x1)*(m<x1) for x0, x1 in zip(starting, ending)]
    part_y = [(l>=y0)*(n>=y0)*(l<y1)*(n<y1) for y0, y1 in zip(starting, ending)]
    F = [torch.exp(2j*np.pi* ((k-x0)*(m-x0)/Lx + (l-y0)*(n-y0)/Ly))/np.sqrt(Lx*Ly)*X*Y for x0, Lx, X in zip(starting, n_list, part_x) for y0, Ly, Y in zip(starting,n_list, part_y)]
    F = torch.stack(F, dim=0).sum(dim=0).reshape(-1, N, N).reshape(-1, N**2)
    filt = np.sort([a*N+b for a in starting for b in starting])
    return F, filt


F, det_filt = {}, {}
F["1"], det_filt["1"] = Fourier_block([28])
# F["5"], det_filt["5"] = Fourier_block([5, 6, 6, 6, 5])
# F["6"], det_filt["6"] = Fourier_block([4,5,5,5,5,4])
F["7"], det_filt["7"] = Fourier_block([4]*7)
# F["8"], det_filt["8"] = Fourier_block([3,3,4,4,4,4,3,3])
# F["9"], det_filt["9"] = Fourier_block([3,3,3,3,4,3,3,3,3])
F["10"], det_filt["10"] = Fourier_block([2,3,3,3,3,3,3,3,3,2])
# F["11"], det_filt["11"] = Fourier_block([3,3,3,2,2,2,2,2,3,3,3])
# F["12"], det_filt["12"] = Fourier_block([2,2,2,2,3,3,3,3,2,2,2,2])
# F["13"], det_filt["13"] = Fourier_block([3,2,2,2,2,2,2,2,2,2,2,2,3])
# F["14"], det_filt["14"] = Fourier_block([2]*14)

def HaarRandU(**kwargs): 
    sd = kwargs["seed"] if "seed" in kwargs else np.random.randint(0, 1e10)
    torch.manual_seed(sd)
    A = torch.normal(torch.zeros(2, N**2,N**2), torch.ones(2, N**2,N**2))
    A = A[0] + 1j* A[1]
    Q, R = torch.linalg.qr(A)
    Lambda = torch.diag( torch.diag(R) / torch.diag(R).abs() )
    return Q@Lambda

# interferometer
# F_inter = torch.cat([torch.cat([Id, 1j*Id],dim=1), torch.cat([1j*Id, Id],dim=1)], dim=0)/np.sqrt(2) @ torch.cat([torch.cat([F, F*0],dim=1), torch.cat([F*0, Id],dim=1)], dim=0)


#%% Fig 2 Network training and model generation

# train_optical_list = [True, True, False, False, False]
# type_init_list = ["F1", "R", "I", "F10", "F7"]

train_optical_list = [True, True, False, False, False, False, False, False, False]
nopool_list = [True, True, True, False, False, True, True, True, True]
type_init_list = ["F1", "R", "I", "F10", "F7", "F1", "R", "F10", "F7"]

for type_init, train_optical, nopool in zip(type_init_list[:3], train_optical_list[:3], nopool_list[:3]):

    if nopool:
        N_filt = 28
        filt = np.arange(784)
    else:
        N_filt = int(type_init[1:])
        filt = det_filt[type_init[1:]]

    if type_init[0] == "F":
        Unitary_init = F[type_init[1:]]
    elif type_init[0] == "R":
        Unitary_init = HaarRandU(seed=0)
    else:
        Unitary_init = Id

    # if len(type_init)>3:
    #     N_filt = 28
    #     filt = np.arange(784)
    # else:
    #     N_filt = int(type_init[1:]) if (not train_optical and type_init[0]=="F") else 28
    #     filt = det_filt[type_init[1:]] if (not train_optical and type_init[0]=="F") else np.arange(784)
    #     Unitary_init = F[type_init[1:]]*(type_init[0]=="F") if type_init[0]=="F" else Id*(type_init[0]=="I") + HaarRandU(seed=0)*(type_init=="R")

    I_sat = 0.5
    p = 0.2
    N_epoch = 2000
    N_modes_optics = [784]*2
    N_modes_digital = [N_filt**2, 300, 200, 50, 10]
    # N_modes_digital = [N_filt**2, 100, 40, 10] # Shallow
    # N_modes_digital = [N_filt**2, 300, 250, 200, 50, 10] # deeper
    batch_size = 5000

    print("Type:{}, Train optical: {}, ".format(type_init, train_optical))

    #### Noise set 1 ####
    # I_dark = I_diff/np.sqrt(2)
    # dt = 1/(I_diff/np.sqrt(2))**2

    #### Noise set 2 ####
    # I_dark = I_diff
    # dt = 1/(I_diff/5)**2

    #### Noise set 3 ####
    # I_dark = I_diff/2
    # dt = 1/(I_diff/10)**2

    #### Noise set 4 ####
    # I_dark = I_diff * 1.5
    # dt = 1e2

    #### Noise set 5 ####
    # I_dark = I_diff * 1.0
    # dt = 1e3

    #### Noise set 6 ####
    # I_dark = I_diff * 0.5
    # dt = 1e4

    #### Noise set 7 ####
    I_dark = I_diff * 0.01
    dt = 1e5


    Noise_tot = np.sqrt(1/dt + I_dark**2)
    SNR_expected = 1/Noise_tot

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    lr = 4e-6 if train_optical else 4e-5

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
    train_loss, val_loss, opt_state_dict = om.run(net, optimizer, criterion, N_epoch, loader, loader_val, device, real=False)
    N_add = 0
    while np.argmin(val_loss) > max(len(val_loss)-100, len(val_loss)*0.85) and N_add<25:
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
    }, "Data_Fig2/net_type{}_train{}_{}.pt".format(type_init,train_optical, "pool" if not nopool else ""))


    net.load_state_dict(opt_state_dict)
    net.eval()
    with torch.no_grad():
        y = net(X_test.to(device)).cpu()
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
    print("Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    net[1].set(dt=(0.01*I_diff)**(-2), dark=(0.01*I_diff))
    with torch.no_grad():
        y = net(X_test.to(device)).cpu()
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
    print("Zero-noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    net[1].set(dt=(I_diff)**(-2), dark=(0.01*I_diff))
    with torch.no_grad():
        y = net(X_test.to(device)).cpu()
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
    print("MAx Shot noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    net[1].set(dt=(0.01*I_diff)**(-2), dark=(I_diff))
    with torch.no_grad():
        y = net(X_test.to(device)).cpu()
        test_infer = torch.argmax(y, dim=1)
    test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
    print("Max dark noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

    fig, ax = plt.subplots(figsize=(2,1.5))
    ax.plot(range(len(train_loss)), train_loss, lw=0.75)
    ax.plot(range(len(train_loss)), val_loss, lw=0.75)
    ax.axvline(x=np.argmin(val_loss), lw=0.5, color='k', ls='--')
    ax.set(xlabel=r'Epoch', ylabel='Cross Entropy Loss', xlim=(-2, len(train_loss)+1))
    ax.set_title("Idiff, Type: {}, trainoptical: {}".format(type_init, train_optical), fontsize=6)

    torch.cuda.empty_cache()



#%%


# net.load_state_dict(torch.load("Data_Fig2_Tdiff30_MNIST/net_typeI_trainFalse.pt")["net_state_dict_opt"])


test_accuracy_dark = []
test_accuracy_shot = []
noise_level = np.linspace(0,2,41) 
noise_level[0] =0.01 


dark_noise_level = np.linspace(0, 0.3, 31)
dt_level = np.exp(np.linspace(np.log(1e4), np.log(0.5), 41))

for repeat in range(10):
    for nl in dark_noise_level:
        net[1].set(dt=(0.01*I_diff)**(-2), dark=nl)
        net.eval()
        with torch.no_grad():
            y = net(X_test.to(device)).cpu()
        test_infer = torch.argmax(y, dim=1)
        test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
        test_accuracy_dark.append(test_accuracy)

    for dt in dt_level:
        net[1].set(dt=dt, dark=0.0)
        net.eval()
        with torch.no_grad():
            y = net(X_test.to(device)).cpu()
        test_infer = torch.argmax(y, dim=1)
        test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
        test_accuracy_shot.append(test_accuracy)

test_accuracy_dark = np.array(test_accuracy_dark).reshape(10,-1)
test_accuracy_shot = np.array(test_accuracy_shot).reshape(10,-1)

fig, ax =plt.subplots(1,2, sharey=True, figsize=(4,2))
ax[0].errorbar(dark_noise_level, test_accuracy_dark.mean(axis=0)*100, yerr=test_accuracy_dark.std(axis=0)*100)
ax[1].errorbar(dt_level, test_accuracy_shot.mean(axis=0)*100, yerr=test_accuracy_shot.std(axis=0)*100)

ax[0].set(xlim=(0,0.3), xlabel=r'Noise power (Arb. U.)', ylim=(10,100), ylabel=r'Test accuracy (%)')
ax[1].set(xlim=(1e4,0.5), xlabel=r'Exposure time (Arb. U.)')
ax[1].set_xscale("log")


fig.tight_layout()