#%% Dataset
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import plot_setting
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn
import optics_module as om
import torchvision.datasets as ds


torch.set_default_device("cuda" if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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
F["7"], det_filt["7"] = om.Fourier_block([4]*7)
F["10"], det_filt["10"] = om.Fourier_block([2,3,3,3,3,3,3,3,3,2])

torch.cuda.empty_cache()

#%% Data calculation for main Fig.5

I_sat = 0.5
p = 0.2

dark_noise_level = np.linspace(0, 2, 51)*I_diff
dt_level = np.exp(np.linspace(np.log(1e4), np.log(1e1), 51))

train_optical_list = [True, True, False, False, False]
type_init_list = ["F1", "R", "I", "F10", "F7"]

test_accuracy = []
test_IPR = []
min_noise_power = []
N_repeat = 20

test_noisy_signal = []
test_pure_signal = []
test_pure_signal_bymean = []

test_pure_prob = []
test_noisy_prob = []

N_phase = 10000
phi = 2*np.pi*torch.rand(N_phase, 1, 784)

for type_init, train_optical in zip(type_init_list, train_optical_list): 

    N_filt = int(type_init[1:]) if (not train_optical and type_init[0]=="F") else 28
    filt = det_filt[type_init[1:]] if (not train_optical and type_init[0]=="F") else np.arange(784)

    N_modes_optics = [784]*2
    N_modes_digital = [N_filt**2, 300, 200, 50, 10]

    checkpoint = torch.load("Data_SuppFig_incoherence/net_type{}_train{}.pt".format(type_init,train_optical), map_location=device)
    print(type_init, train_optical)
    net = om.classifier(
        N_modes_optics, N_modes_digital, I_sat, 1e4, 0.01, p, 
        optical=True, digital=True, batchnormfirst=True, affine=False, filter=filt
    )
    net.load_state_dict(checkpoint["net_state_dict_opt"])
    net.eval()

    with torch.no_grad():
        E_in = X_test.to(device)
        I_in = E_in.abs()**2
        W = (net[0][0].weight.abs()**2).T
        I_out = I_in @ W 
        I_out_detach = I_out.detach()

        E_in_phased = X_test[np.arange(10)*1000].reshape(1,-1, 784).to(device) * torch.exp(1j*phi)
        E_in_phased = E_in_phased.reshape(-1,784)
        I_out_phased = net[0](E_in_phased).abs()**2
        I_out_meanphase = I_out_phased.reshape(N_phase, -1, 784).mean(dim=0)

        test_pure_signal.append(I_out[np.arange(10)*1000].cpu().numpy())
        test_pure_signal_bymean.append(I_out_meanphase.cpu().numpy())
        prob = nn.functional.softmax(net[3](net[2](I_out[np.arange(10)*1000])), dim=1)
        test_pure_prob.append(prob.cpu().numpy())

    ### Calculation accuracy drop
    for noise_type in ["dark", "shot"]:

        for I_dark, dt in zip(dark_noise_level, dt_level):
            net[1].set(dt=(1e4 if noise_type=="dark" else dt), dark=(0 if noise_type=="shot" else I_dark))
            
            for repeat in range(N_repeat):
                with torch.no_grad():
                    poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
                    gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
                    x = I_out + poisson + gaussian
                    y = net[3](net[2](x))
                test_infer = torch.argmax(y, dim=1)
                acc = ((Y_test == test_infer)+0.0).mean().item()
                test_accuracy.append(acc)

            with torch.no_grad():
                test_noisy_signal.append(x[np.arange(10)*1000].cpu().numpy())
                prob = nn.functional.softmax(y[np.arange(10)*1000], dim=1)
                test_noisy_prob.append(prob.cpu().numpy())

test_accuracy = np.array(test_accuracy).reshape(len(type_init_list), 2, len(dt_level), N_repeat)
test_pure_signal = np.array(test_pure_signal).reshape(len(type_init_list), 10, -1)
test_pure_signal_bymean = np.array(test_pure_signal_bymean).reshape(len(type_init_list), 10, -1)
test_noisy_signal = np.array(test_noisy_signal).reshape(len(type_init_list), 2, len(dt_level), 10, -1)

test_pure_prob = np.array(test_pure_prob).reshape(len(type_init_list), 10, -1)
test_noisy_prob = np.array(test_noisy_prob).reshape(len(type_init_list), 2, len(dt_level),  10, -1)

np.savez(
    "Data_SuppFig_incoherence/test.npz",
    darknoise=dark_noise_level,
    shotnoise=dt_level,
    accuracy=test_accuracy, 
    puresignal=test_pure_signal,
    puresignal_bymean=test_pure_signal_bymean,
    noisysignal=test_noisy_signal,
)


#%% Supp Fig - incoherence - unitary

with np.load("Data_SuppFig_incoherence/test.npz") as plot_data:
    dark_noise_level = plot_data["darknoise"]
    test_accuracy = plot_data["accuracy"]
    test_pure_signal = plot_data["puresignal"]
    test_pure_signal_bymean = plot_data["puresignal_bymean"]
    test_noisy_signal = plot_data["noisysignal"]


fig, ax = plt.subplots(2,3, figsize=(6, 2.3))

digit = 0
ax[0,0].matshow(test_pure_signal[2, digit].reshape(28,28).T, cmap=plt.cm.inferno, vmin=0.97, vmax=1.03+I_diff)
ax[0,1].matshow(test_pure_signal[1, digit].reshape(28,28).T, cmap=plt.cm.inferno, vmin=0.97, vmax=1.03+I_diff)
ms = ax[0,2].matshow(test_pure_signal[4, digit].reshape(28,28).T, cmap=plt.cm.inferno, vmin=0.97, vmax=1.03+I_diff)

ax[1,0].matshow(test_pure_signal_bymean[2, digit].reshape(28,28).T, cmap=plt.cm.inferno, vmin=0.97, vmax=1.03+I_diff)
ax[1,1].matshow(test_pure_signal_bymean[1, digit].reshape(28,28).T, cmap=plt.cm.inferno, vmin=0.97, vmax=1.03+I_diff)
ms = ax[1,2].matshow(test_pure_signal_bymean[4, digit].reshape(28,28).T, cmap=plt.cm.inferno, vmin=0.97, vmax=1.03+I_diff)

ax[0,0].set(ylabel=r'$\langle  \mathbf{I}^\mathrm{(out)} \rangle_t$, Theory')
ax[1,0].set(ylabel=r'$\langle  \mathbf{I}^\mathrm{(out)} \rangle_t$, Ensemble')

[ax[jj,ii].set(xticks=(), yticks=()) for jj in range(2) for ii in range(3)]
[ax[jj,ii].spines[pos].set_visible(False) for pos in ["top", "bottom", "left", "right"]  for jj in range(2) for ii in range(3)]

ax[0,0].set_title("Unprocessed Image", fontsize=7, color='k')
ax[0,1].set_title("Random (trained)", fontsize=7, color='darkorange')
ax[0,2].set_title("Block Fourier (fixed)", fontsize=7, color='teal')

axins = ax[0,2].inset_axes([1.03, 0.25, 0.05, 0.5])
cb = plt.colorbar(ms, orientation='vertical', cax=axins, extend=None)
cb.ax.set_yticks([1, 1.2])

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/SuppFig_incoherenceA_{}.pdf".format(time), dpi=900, transparent=True)


fig, ax = plt.subplots(1,2, figsize=(6,2.2))


randinitmat = np.load("HaarRand0.npz")["arr_0"]
Random_init_signal = test_pure_signal[2,digit] @ np.abs(randinitmat)**2
Fourier_init_signal = test_pure_signal[2,digit] @ np.abs(F["1"].cpu().numpy())**2

ax[0].plot(np.sort(Random_init_signal), color='darkorange', lw=0.8)
ax[0].plot(np.sort(Fourier_init_signal), color='crimson', lw=0.8)

ax[0].plot(np.sort(test_pure_signal[2, digit]), lw=1.2, color='k')
ax[0].plot(np.sort(test_pure_signal[0, digit]), lw=1.2, color='crimson', ls='--')
ax[0].plot(np.sort(test_pure_signal[1, digit]), lw=1.2, color='darkorange', ls=':')
ax[0].plot(np.sort(test_pure_signal[3, digit]), lw=1.2, color='royalblue')
ax[0].plot(np.sort(test_pure_signal[4, digit]), lw=1.2, color='teal')
ax[0].set(xlabel=r'Index (sorted), $\alpha$', ylabel=r'Time-averaged intensity, $\langle I_\alpha^\mathrm{(out)}\rangle_t$')
ax[0].set(ylim=(1, 1+I_diff), xlim=(300,784), xticks=(300,400,500,600, 700), xticklabels=[r"$\leq 300$", 400, 500, 600, 700])

ax[0].set_title("Intensity distribution", fontsize=7)

det_area = 10**2 #um2
spectral_window = 1 #um
time_window = 1e-6 # sec
ref_dt = om.photon_flux_per_area(10, torch.tensor([300])).item() * det_area * spectral_window * time_window

ax[1].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[0,0], color='crimson', lw=1.2, ls='--', label="Fourier")
ax[1].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[1,0], color='darkorange', lw=1.2, ls='--', label="Random")
ax[1].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[2,0], color='k', lw=1.2, ls='-', label="Image")
ax[1].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[3,0], color='royalblue', lw=1.2, ls='-', label="BF-10")
ax[1].plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=3)[4,0], color='teal', lw=1.2, ls='-', label="BF-7")

ax[1].fill_between(dark_noise_level/I_diff, 100, 100*test_accuracy.mean(axis=3)[2,0], color='k', alpha=0.2)

ax[1].set(xlim=(0,2), ylim=(50,100), ylabel=r'Test accuracy (%)', xlabel=r'Noise power, $\sigma_\mathrm{dark}/\Delta I$')
ax[1].legend(frameon=False, fontsize=7)

# ax[1].axvline(x=dark_noise_level[25]/I_diff, color='gray', lw=0.75, ls='--')

ax[1].set_title("Dark noise", fontsize=7)

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/SuppFig_incoherenceB_{}.pdf".format(time), dpi=900, transparent=True)

