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
F["1"], det_filt["1"] = om.block_focusing_intensity([28], 0.99)
F["7"], det_filt["7"] = om.block_focusing_intensity([4]*7, 0.99)
F["10"], det_filt["10"] = om.block_focusing_intensity([2,3,3,3,3,3,3,3,3,2], 0.99)

G, det_filt = {}, {}
G["1"], det_filt["1"] = om.block_focusing_intensity([28], 0.5)
G["7"], det_filt["7"] = om.block_focusing_intensity([4]*7, 0.5)
G["10"], det_filt["10"]  = om.block_focusing_intensity([2,3,3,3,3,3,3,3,3,2], 0.5)

torch.cuda.empty_cache()

#%% Data calculation for main Fig.5

I_sat = 0.5
p = 0.2

dark_noise_level = np.linspace(0, 2, 51)*I_diff
dt_level = np.exp(np.linspace(np.log(1e4), np.log(1e1), 51))

train_optical_list = [False, False, False, False, False]
type_init_list = ["I", "F10", "F7", "G10", "G7"]

test_accuracy = []
test_IPR = []
min_noise_power = []
N_repeat = 20

test_noisy_signal = []
test_pure_signal = []

test_pure_prob = []
test_noisy_prob = []

for type_init, train_optical in zip(type_init_list, train_optical_list): 

    N_filt = int(type_init[1:]) if not type_init[0]=="I" else 28
    filt = det_filt[type_init[1:]] if not type_init[0]=="I" else np.arange(784)

    N_modes_optics = [784]*2
    N_modes_digital = [N_filt**2, 300, 200, 50, 10]

    checkpoint = torch.load("Data_Fig5/net_type{}_train{}.pt".format(type_init,train_optical), map_location=device)
    print(type_init, train_optical)
    net = om.classifier(
        N_modes_optics, N_modes_digital, I_sat, 1e4, 0.01, p, 
        optical=True, digital=True, batchnormfirst=True, affine=False, filter=filt, incoherent=(type_init!="I")
    )
    net.load_state_dict(checkpoint["net_state_dict_opt"])
    net.eval()

    with torch.no_grad():
        I_signal = net[0](X_test[np.arange(10)*1000].to(device)).abs()**2
        test_pure_signal.append(I_signal.cpu().numpy())
        prob = nn.functional.softmax(net(X_test[np.arange(10)*1000].to(device)), dim=1)
        test_pure_prob.append(prob.cpu().numpy())

    ### Calculation accuracy drop

    for I_dark, dt in zip(dark_noise_level, dt_level):
        net[1].set(dt=1e4, dark=I_dark)
        
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


test_accuracy = np.array(test_accuracy).reshape(len(type_init_list),  len(dt_level), N_repeat)
test_pure_signal = np.array(test_pure_signal).reshape(len(type_init_list), 10, -1)
test_noisy_signal = np.array(test_noisy_signal).reshape(len(type_init_list),  len(dt_level), 10, -1)

test_pure_prob = np.array(test_pure_prob).reshape(len(type_init_list), 10, -1)
test_noisy_prob = np.array(test_noisy_prob).reshape(len(type_init_list), len(dt_level),  10, -1)

# np.savez(
#     "Data_SuppFig_intensityfocusing/test.npz"
#     darknoise=dark_noise_level,
#     shotnoise=dt_level,
#     accuracy=test_accuracy, 
#     puresignal=test_pure_signal,
#     noisysignal=test_noisy_signal,
# )

#%% Fig. 5

with np.load("Data_SuppFig_intensityfocusing/test.npz") as plot_data:
    dark_noise_level = plot_data["darknoise"]
    dt_level = plot_data["shotnoise"]
    test_accuracy = plot_data["accuracy"]
    test_pure_signal = plot_data["puresignal"] 
    test_noisy_signal = plot_data["noisysignal"]

fig, ax = plt.subplots(figsize=(3.35,2))

ax.plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=2)[1], color='royalblue', lw=0.75, ls='-', label="Focusing: 10 seg")
ax.plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=2)[2], color='teal', lw=0.75, ls='-', label="Focusing: 7 seg")
ax.plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=2)[3], color='royalblue', lw=0.75, ls='--')
ax.plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=2)[4], color='teal', lw=0.75, ls='--')
ax.plot(dark_noise_level/I_diff, 100*test_accuracy.mean(axis=2)[0], color='k', lw=0.75, ls='-', label="Image")

ax.set(xlim=(0,2), ylim=(60,100), ylabel=r'Test accuracy (%)', xlabel=r'Noise power, $\sigma_\mathrm{dark}/\Delta I$')
ax.legend(frameon=False, fontsize=7)

ax.text(1.65, 85, r'50%')
ax.text(1.65, 95, r'99%')

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/SuppFig_intensityfocusing_{}.pdf".format(time), dpi=900, transparent=True)

