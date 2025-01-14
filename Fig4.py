#%% Dataset
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import plot_setting
import copy
from datetime import datetime
import torch
from torch import nn
import optics_module as om
import torchvision.datasets as ds

torch.set_default_device("cuda" if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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
val_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*val_set/255)/ambient)**0.5
test_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient)**0.5
I_diff = test_set.abs().max().item()**2-1

X_val = val_set.reshape(-1,28,28).reshape(-1,28*28)
X_test = test_set.reshape(-1,28,28).reshape(-1,28*28)
Y_test = (torch.arange(N_class).reshape(-1,1) * torch.ones(1,test_set.shape[1])).reshape(-1)
torch.cuda.empty_cache()


#%% Calculate entropy and test accuracy drop upon pruning

I_sat = 0.5
p = 0.2
I_dark = 0.0
dt = 1e4

N_modes_optics = [784]*2
N_modes_digital = [784, 300, 200, 50, 10]

type_init = "R"
train_optical = True
ens = 0

dark_noise_level = np.array([0.01, 0.1, 0.2])
N_prune_list = np.arange(0, 784, 25)


test_accuracy_prune = []
val_entropy = []
val_overlap = []
val_pdf = []

for ens in range(10):
    for jj, train_noise_power in enumerate(dark_noise_level):
        net = om.classifier(
            N_modes_optics, N_modes_digital, I_sat, dt, I_dark, p, 
            optical=True, digital=True, batchnormfirst=True, affine=False
        )
        checkpoint = torch.load("Data_Fig4/net_random_darknoise{}_ens{}.pt".format(train_noise_power, ens), map_location=device)
        net.load_state_dict(checkpoint["net_state_dict_opt"])

        net.eval()
        with torch.no_grad():
            I_signal = net[0](X_val.to(device)).abs()**2
            y = nn.functional.softmax(net(X_test.to(device)), dim=1)
            test_infer = torch.argmax(y, dim=1)
        test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
        print("Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

        dI = 0.2
        nclass = round(10/dI)
        I_signal_norm = (I_signal-I_signal.mean(dim=0,keepdim=True))/I_signal.std(dim=0,keepdim=True)
        hist = nn.functional.one_hot(((I_signal_norm+4)/dI).clip(min=0, max=nclass-0.00001).to(torch.int64), num_classes=nclass).sum(dim=0)
        pdf = hist / hist.sum(dim=1, keepdim=True)/dI
        entp = -dI*(pdf*torch.log2(pdf+1e-8)).sum(dim=1)
        val_entropy.append(entp.cpu().numpy())

        I_signal_norm = I_signal_norm.reshape(10, -1, 784)
        hist = nn.functional.one_hot(((I_signal_norm+4)/dI).clip(min=0, max=nclass-0.00001).to(torch.int64), num_classes=nclass).sum(dim=1)
        pdf = hist / hist.sum(dim=2, keepdim=True)/dI
        pdf = pdf.transpose(0,1)
        val_pdf.append(pdf.cpu().numpy())

        for stype in ["reverse", "entropy"]:
            sorting = torch.argsort(entp, descending=True) if stype=='reverse' else torch.argsort(entp)
            for N_prune in N_prune_list:
                print(jj, stype, N_prune)
                mask = sorting[:N_prune] 
                prunevalue = I_signal.mean(dim=0)[mask]
                net_prune = om.classifier_prune(net, mask, prunevalue) if N_prune>0 else copy.deepcopy(net)
                net_prune.eval()
                with torch.no_grad():
                    y = nn.functional.softmax(net_prune(X_test.to(device)), dim=1)
                    test_infer = torch.argmax(y, dim=1)
                test_accuracy = ((Y_test == test_infer)+0.0).mean().item()
                test_accuracy_prune.append(test_accuracy)
                torch.cuda.empty_cache()

val_entropy = np.array(val_entropy).reshape(-1, len(dark_noise_level), 784)
val_pdf = np.array(val_pdf).reshape(-1, len(dark_noise_level), 784, 10, 50)
test_accuracy_prune = np.array(test_accuracy_prune).reshape(-1, len(dark_noise_level), 2, len(N_prune_list))

np.savez("Data_Fig4/test.npz", 
         noise=dark_noise_level,
         Nprune=N_prune_list,
         entropy=val_entropy, 
         pdf=val_pdf,
         accuracy=test_accuracy_prune, 
         signalnorm=I_signal_norm.cpu().numpy(),
)


#%% Fig. 4b: Pruning plot new

with np.load("Data_Fig4/test.npz") as plot_data:
    dark_noise_level = plot_data["noise"]
    Nprune=N_prune_list = plot_data["Nprune"]
    val_entropy = plot_data["entropy"]
    val_pdf = plot_data["pdf"]
    test_accuracy_prune = plot_data["accuracy"]

fig, ax = plt.subplots(figsize=(3.35,2))

cmap = plt.cm.copper
c1, c2, c3 = cmap([0, 0.4, 0.7])

d0, = ax.plot(N_prune_list, test_accuracy_prune.mean(axis=0)[0, 0]*100, color=c1, ls='--', marker='s', ms=2.2, lw=0.6, mfc='white', mew=0.5)
d1, = ax.plot(N_prune_list, test_accuracy_prune.mean(axis=0)[1, 0]*100, color=c2, ls='--', marker='s', ms=2.2, lw=0.6, mfc='white', mew=0.5)
d2, = ax.plot(N_prune_list, test_accuracy_prune.mean(axis=0)[2, 0]*100, color=c3, ls='--', marker='s', ms=2.2, lw=0.6, mfc='white', mew=0.5)
a0, = ax.plot(N_prune_list, test_accuracy_prune.mean(axis=0)[0, 1]*100, color=c1, ls='-', marker='o', ms=2.2, lw=0.6, mew=0.5)
a1, = ax.plot(N_prune_list, test_accuracy_prune.mean(axis=0)[1, 1]*100, color=c2, ls='-', marker='o', ms=2.2, lw=0.6, mew=0.5)
a2, = ax.plot(N_prune_list, test_accuracy_prune.mean(axis=0)[2, 1]*100, color=c3, ls='-', marker='o', ms=2.2, lw=0.6, mew=0.5)


ax.text(320,60, r'$\sigma_\mathrm{dark}^\mathrm{(tr)}=0.01$', color=c1)
ax.text(510,80, r'$\sigma_\mathrm{dark}^\mathrm{(tr)}=0.1$', color=c2)
ax.text(640,92, r'$\sigma_\mathrm{dark}^\mathrm{(tr)}=0.2$', color=c3)


ax.legend([(d0,d1,d2), (a0,a1,a2)], ["Descending", "Ascending"], 
          frameon=False, fontsize=7,
          numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})

ax.set(xlim=(0,784), xlabel=r'Cumulative pruning number, $N_\mathrm{prune}$', ylim=(5,100), ylabel='Test accuracy (%)')
fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig4b_{}.pdf".format(time), dpi=900, transparent=True)


#%% Fig. 4A: entropy distribution

fig, ax = plt.subplots(1,3, figsize=(3.35,1.2))

vmin, vmax = val_entropy.min(), val_entropy.max()

ens = 0
ax[0].matshow(val_entropy[ens,0].reshape(28,28).T,  cmap=plt.cm.viridis, vmin=vmin,vmax=vmax)
ax[1].matshow(val_entropy[ens,1].reshape(28,28).T,  cmap=plt.cm.viridis, vmin=vmin,vmax=vmax)
ms = ax[2].matshow(val_entropy[ens,2].reshape(28,28).T,  cmap=plt.cm.viridis, vmin=vmin,vmax=vmax)

cmap = plt.cm.copper
ax[0].set_title(r"$\sigma_\mathrm{dark}^\mathrm{(tr)}=0.01$", fontsize=7, color=c1)
ax[1].set_title(r"$\sigma_\mathrm{dark}^\mathrm{(tr)}=0.1$", fontsize=7, color=c2)
ax[2].set_title(r"$\sigma_\mathrm{dark}^\mathrm{(tr)}=0.2$", fontsize=7, color=c3)

ax[0].set(ylabel=r'Entropy')

for trnoise in range(3):
    sortidx = np.argsort(val_entropy[ens, trnoise])
    idx = sortidx[[0,-1]]
    idx2d = np.unravel_index(idx, (28,28))
    ax[trnoise].plot(idx2d[0][0], idx2d[1][0], color='r', lw=0, marker='o', mfc='None', mew=0.75, ms=7)
    ax[trnoise].plot(idx2d[0][1], idx2d[1][1], color='r', lw=0, marker='s', mfc='None', mew=0.75, ms=7)

axins = ax[2].inset_axes([1.05, 0.2, 0.08, 0.6])
cb = plt.colorbar(ms, orientation='vertical', cax=axins, extend=None)
vmin, vmax = val_entropy[ens,-1].min(), val_entropy[ens,-1].max()
# cb.ax.set_yticks([vmin,vmax])
# cb.ax.set_yticklabels(['min', 'max'])
[ax[i].set(xticks=(), yticks=()) for i in range(3)]
fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig4a_{}.pdf".format(time), dpi=900, transparent=True)


