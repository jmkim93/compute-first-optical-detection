#%% Dataset
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import plot_setting
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn
import optics_module as om
import torchvision.datasets as ds


torch.set_default_device("cuda:0" if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

emnist_test = ds.EMNIST(root="emnist", split='digits', download=True, train=False)


N_class = 10
test_set = emnist_test.data[torch.argsort(emnist_test.targets)]+0.0
test_set_base = test_set.reshape(N_class, -1, 28, 28)[:, :1000]
del emnist_test

#### Low contrast and unnormalized input

Tmin = 300
Tmax = 310
ambient = om.photon_num_per_vol(10, torch.tensor(Tmin)).to('cpu')
test_set = 0j+(om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set_base/255)/ambient)**0.5
test_set_sparse = test_set_base/255 + 0j

X_test = test_set.reshape(-1,28,28).reshape(-1,28*28)
X_test_sparse = test_set_sparse.reshape(-1,28,28).reshape(-1,28*28)
Y_test = (torch.arange(N_class).reshape(-1,1) * torch.ones(1,test_set.shape[1])).reshape(-1)

I_diff = (X_test.abs().max()**2).item()-1
I_diff_sparse = (X_test_sparse.abs().max()**2 - X_test_sparse.abs().min()**2).item()


F, det_filt = {}, {}
F["1"], det_filt["1"] = om.Fourier_block([28])
# F["5"], det_filt["5"] = Fourier_block([5, 6, 6, 6, 5])
# F["6"], det_filt["6"] = Fourier_block([4,5,5,5,5,4])
F["7"], det_filt["7"] = om.Fourier_block([4]*7)
# F["8"], det_filt["8"] = Fourier_block([3,3,4,4,4,4,3,3])
# F["9"], det_filt["9"] = Fourier_block([3,3,3,3,4,3,3,3,3])
F["10"], det_filt["10"] = om.Fourier_block([2,3,3,3,3,3,3,3,3,2])

torch.cuda.empty_cache()

#%% Data calculation for main Fig.2

I_sat = 0.5
p = 0.2

dark_noise_level = np.linspace(0, 2, 51)*I_diff
dt_level = np.exp(np.linspace(np.log(1e5), np.log(1e1), 51))

# train_optical_list = [True, True, False, False, False]
# type_init_list = ["F1", "R", "I", "F10", "F7"]

# train_optical_list = [True, True, False, False, False, False, False, False]
# type_init_list = ["F1", "R", "I", "F10", "F7", "F1_nopool", "R", "F7_nopool"]

train_optical_list = [True, True, False, False, False, False, False, False, False]
nopool_list = [True, True, True, False, False, True, True, True, True]
type_init_list = ["F1", "R", "I", "F10", "F7", "F1", "R", "F10", "F7"]

test_accuracy = []
test_IPR = []
min_noise_power = []
N_repeat = 20

test_noisy_signal = []
test_pure_signal = []

test_pure_prob = []
test_noisy_prob = []

for type_init, train_optical, nopool in zip(type_init_list, train_optical_list, nopool_list): 

    # if len(type_init)>3:
    #     N_filt = 28
    #     filt = np.arange(784)
    # else:
    #     N_filt = int(type_init[1:]) if (not train_optical and type_init[0]=="F") else 28
    #     filt = det_filt[type_init[1:]] if (not train_optical and type_init[0]=="F") else np.arange(784)
    
    if nopool:
        N_filt = 28
        filt = np.arange(784)
    else:
        N_filt = int(type_init[1:])
        filt = det_filt[type_init[1:]]

    N_modes_optics = [784]*2
    N_modes_digital = [N_filt**2, 300, 200, 50, 10]

    checkpoint = torch.load("Data_Fig2/net_type{}_train{}_{}.pt".format(type_init,train_optical, "pool" if not nopool else ""), map_location=device)
    print(type_init, train_optical)
    net = om.classifier(
        N_modes_optics, N_modes_digital, I_sat, 1e4, 0.01, p, 
        optical=True, digital=True, batchnormfirst=True, affine=False, filter=filt
    )
    net.load_state_dict(checkpoint["net_state_dict_opt"])
    net.eval()

    with torch.no_grad():
        I_signal = net[0](X_test[np.arange(10)*1000].to(device)).abs()**2
        test_pure_signal.append(I_signal.cpu().numpy())
        prob = nn.functional.softmax(net(X_test[np.arange(10)*1000].to(device)), dim=1)
        test_pure_prob.append(prob.cpu().numpy())

    ### Calculation accuracy drop
    for noise_type in ["dark", "shot"]:

        for I_dark, dt in zip(dark_noise_level, dt_level):
            net[1].set(dt=(1e5 if noise_type=="dark" else dt), dark=(0 if noise_type=="shot" else I_dark))
            
            for repeat in range(N_repeat):
                with torch.no_grad():
                    y = net(X_test.to(device))
                test_infer = torch.argmax(y, dim=1)
                acc = ((Y_test == test_infer)+0.0).mean().item()
                test_accuracy.append(acc)

            with torch.no_grad():
                I_signal = net[1](net[0](X_test[np.arange(10)*1000].to(device)))
                test_noisy_signal.append(I_signal.cpu().numpy())
                prob = nn.functional.softmax(net(X_test[np.arange(10)*1000].to(device)), dim=1)
                test_noisy_prob.append(prob.cpu().numpy())

test_accuracy = np.array(test_accuracy).reshape(len(type_init_list), 2, len(dt_level), N_repeat)
test_pure_signal = np.array(test_pure_signal).reshape(len(type_init_list), 10, -1)
test_noisy_signal = np.array(test_noisy_signal).reshape(len(type_init_list), 2, len(dt_level), 10, -1)

test_pure_prob = np.array(test_pure_prob).reshape(len(type_init_list), 10, -1)
test_noisy_prob = np.array(test_noisy_prob).reshape(len(type_init_list), 2, len(dt_level),  10, -1)

np.savez(
    "Data_Fig2/test.npz",
    darknoise=dark_noise_level,
    shotnoise=dt_level,
    accuracy=test_accuracy, 
    puresignal=test_pure_signal,
    noisysignal=test_noisy_signal,
)



#%% Fig. 2A

with np.load("Data_Fig2/test.npz") as plot_data:
    dark_noise_level = plot_data["darknoise"]
    dt_level = plot_data["shotnoise"]
    test_accuracy = plot_data["accuracy"]
    test_pure_signal = plot_data["puresignal"] 
    test_noisy_signal = plot_data["noisysignal"]

cmap1 = plt.cm.inferno
cmap2 = plt.cm.cividis

fig, ax = plt.subplots(4, 4, figsize=(4.3, 4), tight_layout=True)
digit = 0
ms00 = ax[0,0].matshow(test_pure_signal[2,digit].reshape(28,28).T, cmap=cmap1, vmin=0.9, vmax=1.4)
ax[2,0].matshow(test_noisy_signal[2,0,25,digit].reshape(28,28).T, cmap=cmap2)
ax[3,0].matshow(test_noisy_signal[2,1,25,digit].reshape(28,28).T, cmap=cmap2)

randinitmat = np.load("HaarRand0.npz")["arr_0"]
Random_init_signal = np.abs(np.sqrt(test_pure_signal[2,digit]+0j) @ randinitmat)**2
ms01 = ax[0,1].matshow(Random_init_signal.reshape(28,28).T, cmap=cmap1, vmin=0, vmax=7)
ms11 = ax[1,1].matshow(test_pure_signal[1,digit].reshape(28,28).T, cmap=cmap1, vmin=0, vmax=7)
ax[2,1].matshow(test_noisy_signal[1,0,25,digit].reshape(28,28).T, cmap=cmap2)
ax[3,1].matshow(test_noisy_signal[1,1,25,digit].reshape(28,28).T, cmap=cmap2)

Fourier_signal = np.abs(np.sqrt(test_pure_signal[2,digit]) @ F["1"].cpu().numpy())**2
ms02 = ax[0,2].matshow(Fourier_signal.reshape(28,28).T, cmap=cmap1, vmin=0, vmax=0.006)
ms12 = ax[1,2].matshow(test_pure_signal[0,digit].reshape(28,28).T, cmap=cmap1, vmin=0, vmax=11)
ax[2,2].matshow(test_noisy_signal[0,0,25,digit].reshape(28,28).T, cmap=cmap2)
ax[3,2].matshow(test_noisy_signal[0,1,25,digit].reshape(28,28).T, cmap=cmap2)


vmin=test_noisy_signal[4,1,15,digit, det_filt["7"]].min()
vmax=test_noisy_signal[4,1,15,digit, det_filt["7"]].max()

ms03 = ax[0,3].matshow(test_pure_signal[4,digit].reshape(28,28).T, cmap=cmap1, vmin=15, vmax=20)
ms13 = ax[1,3].matshow(test_pure_signal[4,digit, det_filt["7"]].reshape(7,7).T, cmap=cmap1, vmin=15, vmax=20)
ms23 = ax[2,3].matshow(test_noisy_signal[4,0,25,digit,  det_filt["7"]].reshape(7,7).T, cmap=cmap2)
ms33 = ax[3,3].matshow(test_noisy_signal[4,1,25,digit, det_filt["7"]].reshape(7,7).T, cmap=cmap2, vmin=vmin,vmax=vmax)


axins00 = ax[0,0].inset_axes([1.03, 0.25, 0.05, 0.5])
cb00 = plt.colorbar(ms00, orientation='vertical', cax=axins00, extend=None)
cb00.ax.set_yticks([1, 1.2])

axins01 = ax[0,1].inset_axes([1.03, 0.25, 0.05, 0.5])
cb01 = plt.colorbar(ms01, orientation='vertical', cax=axins01, extend=None)
cb01.ax.set_yticks([0, 5])

axins11 = ax[1,1].inset_axes([1.03, 0.25, 0.05, 0.5])
cb11 = plt.colorbar(ms11, orientation='vertical', cax=axins11, extend=None)
cb11.ax.set_yticks([0,5])


axins02 = ax[0,2].inset_axes([1.03, 0.25, 0.05, 0.5])
cb02 = plt.colorbar(ms02, orientation='vertical', cax=axins02, extend='max')
cb02.ax.set_yticks([0, 0.005])
cb02.ax.set_yticklabels([0, r'5e-3'])

axins12 = ax[1,2].inset_axes([1.03, 0.25, 0.05, 0.5])
cb12 = plt.colorbar(ms12, orientation='vertical', cax=axins12, extend='max')
cb12.ax.set_yticks([0,5,10])

axins03 = ax[0,3].inset_axes([1.03, 0.25, 0.05, 0.5])
cb03 = plt.colorbar(ms03, orientation='vertical', cax=axins03, extend='min')
cb03.ax.set_yticks([16,18,20])

axins13 = ax[1,3].inset_axes([1.03, 0.25, 0.05, 0.5])
cb13 = plt.colorbar(ms13, orientation='vertical', cax=axins13, extend=None)
cb13.ax.set_yticks([16,18,20])

axins23 = ax[2,3].inset_axes([1.03, 0.25, 0.05, 0.5])
cb23 = plt.colorbar(ms23, orientation='vertical', cax=axins23, extend=None)
cb23.ax.set_yticks([vmin, vmax])
cb23.ax.set_yticklabels(["min", "max"])

axins33 = ax[3,3].inset_axes([1.03, 0.25, 0.05, 0.5])
cb33 = plt.colorbar(ms33, orientation='vertical', cax=axins33, extend=None)
cb33.ax.set_yticks([vmin, vmax])
cb33.ax.set_yticklabels(["min", "max"])

ax[0,0].set_title(r'Ideal image', color='k', fontsize=7)
ax[0,1].set_title(r'Random (fixed)', color='darkorange', fontsize=7)
ax[0,2].set_title(r'Fourier (fixed)', color='crimson', fontsize=7)
ax[1,1].set_title(r'Random (trained)', color='darkorange', fontsize=7)
ax[1,2].set_title(r'Fourier (trained)', color='crimson', fontsize=7)
ax[0,3].set_title(r'Block-Fourier', color='lightseagreen', fontsize=7)
ax[1,3].set_title(r'BF (Pooled)', color='lightseagreen', fontsize=7)
# ax[2,0].set_title("+ Dark noise", fontsize=6.5)
# ax[3,0].set_title("+ Shot noise", fontsize=6.5)

ax[0,3].plot([11.5, 15.5, 15.5, 11.5, 11.5], [11.5, 11.5, 15.5, 15.5, 11.5], color='w', lw=0.4)
# [ax[0,3].axhline(y=4*k-0.5, color='w', lw=0.4, ls='--') for k in range(3,5)]
# [ax[0,3].axvline(x=4*k-0.5, color='w', lw=0.4, ls='--') for k in range(3,5)]
[ax[i,j].set(xticks=(), yticks=()) for i in range(4) for j in range(4)]
[ax[i,j].spines[pos].set_visible(False) for i in range(4) for j in range(4) for pos in ["top", "bottom", "left", "right"]]

time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig2a_{}.pdf".format(time), dpi=900, transparent=True)



#%% Fig. 2B

fig, ax = plt.subplots(2, 1, figsize=(2.3,4))

det_area = 10**2 #um2
spectral_window = 1 #um
time_window = 1e-6 # sec
ref_dt = om.photon_flux_per_area(10, torch.tensor([300])).item() * det_area * spectral_window * time_window


ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[0,0], color='crimson', lw=0.75, ls='-', label="Fourier")
ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[1,0], color='darkorange', lw=0.75, ls='-', label="Random")
ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[4,0], color='lightseagreen', lw=0.75, ls='-', label="BF (pooled)")
ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[2,0], color='k', lw=0.75, ls='--', label="Image")
# ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[3,0], color='royalblue', lw=0.75, ls='-', label="BF-10")
ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[5,0], color='crimson', lw=0.75, ls='--', label="Fourier")
ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[6,0], color='darkorange', lw=0.75, ls='--', label="Random")
ax[0].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[8,0], color='lightseagreen', lw=0.75, ls='--', label="BF")

ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[0,1], color='crimson', lw=0.75, ls='-', label="Fourier")
ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[1,1], color='darkorange', lw=0.75, ls='-', label="Random")
ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[4,1], color='lightseagreen', lw=0.75, ls='-', label="BF (pooled)")
ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[2,1], color='k', lw=0.75, ls='--', label="Image")
# ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[3,1], color='royalblue', lw=0.75, ls='-', label="BF-10")
ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[5,1], color='crimson', lw=0.75, ls='--', label="Fourier")
ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[6,1], color='darkorange', lw=0.75, ls='--', label="Random")
ax[1].plot(dt_level, 100*test_accuracy.mean(axis=3)[8,1], color='lightseagreen', lw=0.75, ls='--', label="BF")


ax[0].set(xlim=(0,2), ylim=(10,100), ylabel=r'Test accuracy (%)', xlabel=r'Noise power, $\sigma_\mathrm{dark}/\Delta I$')
ax[1].set(xlim=(1e5,1e1), ylim=(10,100),ylabel=r'Test accuracy (%)', xlabel=r'Exposure time, $\Delta t$ (Arb. U.)')
ax[1].set_xscale("log")

ax[1].legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.02,0.07))
[ax[i].spines[pos].set_visible(False) for i in range(2) for pos in ["top", "right"]]

I_dark_train = I_diff * 1.0
dt_train = 1e3

ax[0].add_patch(Rectangle((I_dark_train/I_diff, 10), 2-I_dark_train/I_diff, 90, facecolor='k', alpha=0.1))
ax[1].add_patch(Rectangle((dt_train, 10), 1-dt_train, 90, facecolor='k', alpha=0.1))

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig2b_{}.pdf".format(time), dpi=900, transparent=True)




#%% Supplementary Figs: Learning curves

train_optical_list = [False, False, False,  False, True, True,  False]
nopool_list = [True, True, True,  True, True, True,  False]
type_init_list = ["F1", "R",  "F7", "I", "F1", "R",  "F7"]

fig, ax = plt.subplots(2,4, figsize=(6.5,3.3))
ax = ax.reshape(-1)
label="abcdefg"

for ii, (type_init, train_optical, nopool) in enumerate(zip(type_init_list, train_optical_list, nopool_list)): 
    checkpoint = torch.load("Data_Fig2_rev/net_type{}_train{}_{}.pt".format(type_init,train_optical, "pool" if not nopool else ""), map_location=device)
    tl, vl = checkpoint["train_loss"], checkpoint["val_loss"]
    ax[ii].plot(range(len(tl)), tl, lw=0.75, label='Training')
    ax[ii].plot(range(len(tl)), vl, lw=0.75, label='Validation')
    ax[ii].axvline(x=np.argmin(vl), color='k', lw=0.75, ls='--')
    ax[ii].set(xlim=(-5, len(tl)+5), ylim=(0.2,2.5), 
               xlabel=r'Epoch' if ii>2 else '', 
               ylabel=r'Cross-entropy loss' if ii in [0,4] else '')

    # ax[ii].text(len(tl)*0.03, 2.3, label[ii], fontsize=8, fontweight="bold")

ax[3].legend(frameon=False, fontsize=7, loc=1)
ax[-1].set(xticks=(), yticks=())
[ax[-1].spines[pos].set_visible(False) for pos in ["top", "bottom", "left", "right"]]

ax[0].set_title("Fourier (fixed)", fontsize=7, color='crimson')
ax[1].set_title("Random (fixed)", fontsize=7, color='darkorange')
ax[2].set_title("BF", fontsize=7, color='lightseagreen')
ax[3].set_title("Ideal image", fontsize=7)
ax[4].set_title("Fourier (trained)", fontsize=7, color='crimson')
ax[5].set_title("Random (trained)", fontsize=7, color='darkorange')
ax[6].set_title("BF (pooled)", fontsize=7, color='lightseagreen')

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/SuppFig_LearningCurves_{}.pdf".format(time), dpi=900, transparent=True)
