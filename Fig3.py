#%% Dataset
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import plot_setting

from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_default_device("cuda" if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

import optics_module as om
import torchvision.datasets as ds


emnist_train = ds.EMNIST(root="emnist", split='digits', download=True, train=True)
emnist_test = ds.EMNIST(root="emnist", split='digits', download=True, train=False)

N_class = 10
train_set = emnist_train.data[torch.argsort(emnist_train.targets)]+0.0
train_set = train_set.reshape(N_class, -1, 28, 28)
train_set, val_set = train_set[:,:5000], train_set[:,5000:6000]
test_set = emnist_test.data[torch.argsort(emnist_test.targets)]+0.0
test_set = test_set.reshape(N_class, -1, 28, 28)[:, :1000]
del emnist_test, emnist_train, train_set


#### Low contrast and unnormalized input
Tmin = 300
Tmax = 310
ambient = om.photon_num_per_vol(10, torch.tensor(Tmin)).to('cpu')
test_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient)**0.5
val_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*val_set/255)/ambient)**0.5
I_diff = test_set.abs().max().item()**2-1

X_val = val_set.reshape(-1,28,28).reshape(-1,28*28)
X_test = test_set.reshape(-1,28,28).reshape(-1,28*28)
Y_test = (torch.arange(N_class).reshape(-1,1) * torch.ones(1,test_set.shape[1])).reshape(-1)


F, det_filt = {}, {}
F["1"], det_filt["1"] = om.Fourier_block([28])
F["2"], det_filt["2"] = om.Fourier_block([14,14])
F["3"], det_filt["3"] = om.Fourier_block([9,10,9])
F["4"], det_filt["4"] = om.Fourier_block([7]*4)
F["5"], det_filt["5"] = om.Fourier_block([5, 6, 6, 6, 5])
F["6"], det_filt["6"] = om.Fourier_block([4,5,5,5,5,4])
F["7"], det_filt["7"] = om.Fourier_block([4]*7)
F["8"], det_filt["8"] = om.Fourier_block([3,3,4,4,4,4,3,3])
F["9"], det_filt["9"] = om.Fourier_block([3,3,3,3,4,3,3,3,3])
F["10"], det_filt["10"] = om.Fourier_block([2,3,3,3,3,3,3,3,3,2])
F["11"], det_filt["11"] = om.Fourier_block([3,3,3,2,2,2,2,2,3,3,3])
F["12"], det_filt["12"] = om.Fourier_block([2,2,2,2,3,3,3,3,2,2,2,2])
F["13"], det_filt["13"] = om.Fourier_block([3,2,2,2,2,2,2,2,2,2,2,2,3])
F["14"], det_filt["14"] = om.Fourier_block([2]*14)
torch.cuda.empty_cache()


#%% Data calculation

test_accuracy = []
test_signal = []
entropy = []
N_repeat = 20

n_seg_list = np.arange(1,15)
for n in n_seg_list:
    type_init, train_optical = "F{}".format(n), False
    print("Type:{}, Train optical: {}, ".format(type_init, train_optical))

    filt = det_filt[str(n)] if (not train_optical and type_init[0]=="F") else np.arange(784)
    N_modes_digital = [n**2, 300, 200, 50, 10]

    checkpoint = torch.load("Data_Fig3/net_type{}_train{}.pt".format(type_init,train_optical), map_location=device)
    net = om.classifier(
        [784]*2, N_modes_digital, 0.5, 1e4, 0, 0.2, 
        optical=True, digital=True, batchnormfirst=True, affine=False, filter=filt
    )
    net.load_state_dict(checkpoint["net_state_dict_opt"])
    
    for I_dark in np.arange(0, 1.05, 0.1):
        net[1].set(dt=1e4, dark=I_dark)
        net.eval()
        for repeat in range(N_repeat):
            with torch.no_grad():
                y = net(X_test.to(device))
            test_infer = torch.argmax(y, dim=1)
            acc = ((Y_test == test_infer)+0.0).mean().item()
            test_accuracy.append(acc)

    net[1].set(dt=1e4, dark=0)
    dI = 0.1
    nclass = round(20/dI)
    with torch.no_grad():
        I_signal = net[0](X_test[np.arange(0,10000,1000)].to(device)).abs()**2
        I_filt = net[2](net[1](net[0](X_val.to(device))))
    I_filt = (I_filt-I_filt.mean(dim=0, keepdim=True))/I_filt.std(dim=0, keepdim=True)
    hist = nn.functional.one_hot(((I_filt+10)/dI).clip(min=0, max=nclass-0.00001).to(torch.int64), num_classes=nclass).sum(dim=0)
    pdf = hist / hist.sum(dim=1, keepdim=True)/dI
    entp = -dI*(pdf*torch.log2(pdf+1e-8)).sum(dim=1).cpu().numpy()
    entropy.append(entp)
    test_signal.append(I_signal.cpu().numpy())

test_accuracy = np.array(test_accuracy).reshape(len(n_seg_list), -1, N_repeat)
test_signal = np.array(test_signal).reshape(len(n_seg_list), 10, -1)

entropy_dict = {"entropy{}".format(i): entropy[i-1] for i in n_seg_list}
filt_dict = {"filt{}".format(i): det_filt[str(i)] for i in n_seg_list}

np.savez(
    "Data_Fig3/test.npz",
    N_seg = n_seg_list,
    accuracy=test_accuracy, 
    signal=test_signal,
    **entropy_dict,
    **filt_dict
)




#%% Fig. 3A

with np.load("Data_Fig3/test.npz") as plot_data:
    test_accuracy = plot_data["accuracy"]
    test_signal = plot_data["signal"]
    n_seg_list = plot_data["N_seg"]
    entropy = [plot_data["entropy{}".format(i)] for i in n_seg_list]
    det_filt = {str(i): plot_data["filt{}".format(i)] for i in n_seg_list}

ent_max = np.max([entropy[i].max() for i in range(14)])
ent_min = np.min([entropy[i].min() for i in range(14)])

fig, ax = plt.subplots(3, 5, figsize=(3.35,2))
digit = 0
for i, N_seg in enumerate([2,4, 7,10,13]):
    I_signal = test_signal[N_seg-1, digit]
    border = det_filt[str(N_seg)][1:N_seg]
    ms0 = ax[0,i].matshow(I_signal.reshape(28,28).T, vmin=0, vmax=I_signal.max(), cmap=plt.cm.inferno)
    # if i<3:
        # [ax[0,i].axhline(y=b-0.5, lw=0.3, ls='-', color='w') for b in border]
        # [ax[0,i].axvline(x=b-0.5, lw=0.3, ls='-', color='w') for b in border]
    ms1 = ax[1,i].matshow(I_signal[det_filt[str(N_seg)]].reshape(N_seg,-1).T, vmin=I_signal[det_filt[str(N_seg)]].min(), vmax=I_signal[det_filt[str(N_seg)]].max(), cmap=plt.cm.inferno)
    ax[0,i].set_title(r"$N_\mathrm{seg}="+str(N_seg)+"$", fontsize=6)
    ms2 = ax[2,i].matshow(entropy[N_seg-1].reshape(N_seg,-1).T, vmin=ent_min, vmax=ent_max, cmap=plt.cm.viridis)



axins04 = ax[0,4].inset_axes([1.05, 0.2, 0.08, 0.6])
cb04 = plt.colorbar(ms0, orientation='vertical', cax=axins04, extend=None)
cb04.ax.set_yticks([0, I_signal.max()])
cb04.ax.set_yticklabels([0, 'max'])


axins14 = ax[1,4].inset_axes([1.05, 0.2, 0.08, 0.6])
cb14 = plt.colorbar(ms1, orientation='vertical', cax=axins14, extend=None)
cb14.ax.set_yticks([I_signal[det_filt[str(N_seg)]].min(), I_signal[det_filt[str(N_seg)]].max()])
cb14.ax.set_yticklabels(['min', 'max'])

axins24 = ax[2,4].inset_axes([1.05, 0.2, 0.08, 0.6])
cb24 = plt.colorbar(ms2, orientation='vertical', cax=axins24, extend=None)
cb24.ax.set_yticks([1, 2])

ax[0,0].set(ylabel=r'BF')
ax[1,0].set(ylabel=r'Pooled')
ax[2,0].set_ylabel(r'Entropy', color='seagreen')

[ax[i,j].set(xticks=(), yticks=()) for i in range(3) for j in range(5)]
[ax[i,j].spines[pos].set_visible(False) for i in range(2) for j in range(5) for pos in ["top", "bottom", "left", "right"]]
fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig3a_{}.pdf".format(time), dpi=900, transparent=True)

#%% Fig. 3B 

fig, ax = plt.subplots(figsize=(3.35,2))

cmap = plt.cm.copper

for i in range(11):
    ax.plot(n_seg_list, test_accuracy.mean(axis=2)[:,i]*100, 'o-', color=cmap(i/10.01), lw=0.7, ms=1.2)

ax.set(
    xlabel=r'Segmentation number, $N_\mathrm{seg}$', ylabel=r'Test accuracy (%)',
    xlim=(1,14), ylim=(20,100)
)

axt = ax.twinx()
mean_entropy = [entp.mean() for entp in entropy]
axt.plot(n_seg_list, mean_entropy, 
         'o-', color='seagreen', lw=0.75, 
         ms=2.5, markerfacecolor='white', markeredgewidth=0.5
)
axt.set_ylabel(r'Entropy (bit/pixel)', color='seagreen')
axt.spines["right"].set_edgecolor('seagreen')
axt.tick_params(axis='y', colors='seagreen')

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig3b_{}.pdf".format(time), dpi=900, transparent=True)


