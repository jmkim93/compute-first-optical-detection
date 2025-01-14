#%% Dataset

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


import torch
from  torch.nn.modules.upsampling import Upsample
# from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
torch.set_default_device("cuda:3")
device = torch.device("cuda:3")
import torchvision.datasets as ds

import optics_module as om


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
train_set_intensity = om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*train_set/255)/ambient
val_set_intensity = om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*val_set/255)/ambient
test_set_intensity = om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient
torch.cuda.empty_cache()
I_diff = (train_set_intensity.max()-train_set_intensity.min()).item()

### Wave diffraction definition
wvl = 1.0
k0 = 2*np.pi/wvl
# resol_scene = 80 # including padded region
# resol_out = 40
# f = 300
# dx = 2.0

# f = 200
# dx = 1.0

resol_scene = 60 # including padded region
resol_out = 28
f = 300
dx = 1.5

W = resol_scene * dx
W_lens = 3*W/2
mag = 1
resol_lens = int(W_lens/dx)

NA = np.sin(np.arctan(W_lens/(2*f)))
crop_a, crop_b = int((resol_scene-resol_out)/2), int((resol_scene+resol_out)/2)
upsample = Upsample(size=(resol_out,resol_out), mode='bilinear')

def dataset_field(intensity):
    intensity_up = intensity if resol_out==28 else upsample(intensity)
    temp = torch.ones(*intensity_up.shape[:2], resol_scene,resol_scene)
    temp[:,:, crop_a:crop_b, crop_a:crop_b] = intensity_up
    torch.cuda.empty_cache()
    return 0j + torch.sqrt(temp).reshape(-1, resol_scene, resol_scene)

train_set = dataset_field(train_set_intensity).cpu()
val_set = dataset_field(val_set_intensity).cpu()
test_set = dataset_field(test_set_intensity).cpu()

del train_set_intensity, val_set_intensity, test_set_intensity
torch.cuda.empty_cache()

train_optical = False

resol_optics = [resol_scene, resol_out]
N_modes_digital = [resol_out**2, 300, 200, 50, 10]
p = 0.2

net = om.classifier_diffraction(resol_optics, W, W_lens, f, wvl, N_modes_digital, 1e5, 0, p,
                                incoherent=True, train=train_optical, batchnormfirst=True, affine=False, magnify=mag, Ndx=5)
with torch.no_grad():
    E_in = test_set[::1000].reshape(10, -1).to(device)
    E_out = net[0](E_in)
    I_diff_image = (E_out.abs().max()**2 - E_out.abs().min()**2).item()

I_dark = I_diff_image*0.25  # Quarternoise. 0.5 for halfnoise
dt = 1e6
net[1].set(dt=dt, dark=I_dark)

fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
ax[0,0].matshow(E_in[0].reshape(resol_scene,-1)[crop_a:crop_b, crop_a:crop_b].abs().cpu().T**2)
ax[1,0].matshow(E_out[0].reshape(resol_out,-1).abs().cpu().T**2)
ax[0,1].matshow(E_in[5].reshape(resol_scene,-1)[crop_a:crop_b, crop_a:crop_b].abs().cpu().T**2)
ax[1,1].matshow(E_out[5].reshape(resol_out,-1).abs().cpu().T**2)

fig.tight_layout()
fig, ax = plt.subplots(2,1)
ax[0].matshow(net[0].lens1.reshape(resol_lens,-1).cpu()%(2*np.pi), vmin=0, vmax=2*np.pi, cmap=plt.cm.twilight)
ax[1].matshow(net[0].lens3.reshape(resol_lens,-1).cpu()%(2*np.pi), vmin=0, vmax=2*np.pi, cmap=plt.cm.twilight)

print("NA:", NA)
print("I_diff image:", I_diff_image)


#%% Training

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
lr = 1e-4
N_epoch = 2000
batch_size = 5000

##### Dataset #####
X = train_set.reshape(-1,resol_scene**2)
Y = ((torch.arange(N_class).reshape(-1,1) * torch.ones(1,5000)).reshape(-1,1) == torch.arange(N_class).reshape(1,-1) ) + 0.0
X_val = val_set.reshape(-1,resol_scene**2)
Y_val = ((torch.arange(N_class).reshape(-1,1) * torch.ones(1,1000)).reshape(-1,1) == torch.arange(N_class).reshape(1,-1) )+0.0

dataset = TensorDataset(X, Y)
dataset_val = TensorDataset(X_val, Y_val)
X_test = test_set.reshape(-1,resol_scene**2)
Y_test = (torch.arange(N_class).reshape(-1,1) * torch.ones(1,1000)).reshape(-1).cpu()

loader = DataLoader(dataset, batch_size = batch_size, shuffle=True, generator=torch.Generator(device=device))
loader_val = DataLoader(dataset_val, batch_size=10000, shuffle=False, generator=torch.Generator(device=device))
optimizer = torch.optim.Adam((net if train_optical else net[-1]).parameters(), lr=lr)


note = "f={}, dx={}, train_optical={}, LR={}, EpochTot={}".format(f, dx, train_optical, lr,  N_epoch)
writer = SummaryWriter(comment=note)
train_loss, val_loss, opt_state_dict = om.run(
    net, optimizer, criterion, N_epoch, loader, loader_val, device, 
    incoherent=True, nonunitary=True, tensorboard=True, writer=writer
)
N_add = 0

while (np.argmin(val_loss)>len(val_loss)*0.80) and N_add<80:
    N_add += 1
    N_epoch_add = 100
    train_loss, val_loss, opt_state_dict = om.run(
        net, optimizer, criterion, N_epoch_add, loader, loader_val, device,
        resume=True, incoherent=True, nonunitary=True, tensorboard=True, writer=writer,
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
}, "Data_Fig5/net_diff_quarternoise_f{}_train{}.pt".format(f, train_optical))

#%% Evaluation
plt.plot(train_loss)
plt.plot(val_loss)

net.load_state_dict(opt_state_dict)
net.eval()
with torch.no_grad():
    E_in = X_test.to(device)
    y = net(E_in)
    test_infer = torch.argmax(y, dim=1)
test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
print("Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

net[1].set(dt=(0.01*I_dark)**(-2), dark=(0.01*I_dark))
with torch.no_grad():
    E_in = X_test.to(device)
    y = net(E_in)
    test_infer = torch.argmax(y, dim=1)
test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
print("Zero-noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

net[1].set(dt=(I_dark)**(-2), dark=(0.01*I_dark))
with torch.no_grad():
    E_in = X_test.to(device)
    y = net(E_in)
    test_infer = torch.argmax(y, dim=1)
test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
print("MAx Shot noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

net[1].set(dt=(0.01*I_dark)**(-2), dark=(I_dark))
with torch.no_grad():
    E_in = X_test.to(device)
    y = net(E_in)
    test_infer = torch.argmax(y, dim=1)
test_accuracy = ((Y_test.to(device) == test_infer)+0.0).mean().item()
print("Max dark noise Test accuracy: {}%".format( round(test_accuracy*100, 4) ))

torch.cuda.empty_cache()



#%% mask plot

mask = (net[0].amp_activation()*torch.exp(1j*(net[0].phase-net[0].lens))).reshape(resol_lens,-1).detach().cpu()
plt.matshow(mask.angle())


#%% TEST- demonstration

wvl = 1
k0 = 2*np.pi/wvl

dx = 2
resol_sc = 80
W = resol_sc * dx

resol_in = 40


test = test_set[:,0].abs()**2 #intensity
upsample = Upsample(size=(resol_in,resol_in), mode='bilinear')
test = upsample(test.reshape(test.shape[0],-1,*test.shape[1:]))

test_pad = torch.ones(10,1, resol_sc, resol_sc)
test_pad[:,:, int((resol_sc-resol_in)/2):int((resol_sc+resol_in)/2), int((resol_sc-resol_in)/2):int((resol_sc+resol_in)/2)] = test
test = test_pad.clone()


f = 300
mag = 1
H1, H2 = (mag+1)/mag * f, (mag+1) * f
W_lens = 3*W/2
resol_lens = int(W_lens/dx)
NA = np.sin(np.arctan(W_lens/(2*f)))

print("NA:", NA)

net_diff = om.optical_diffraction([resol_sc, resol_in], W, W_lens, f, wvl, 
                                  imaging=True, incoherent=True, Ndx=5, device=device,)

with torch.no_grad():
    E_in = torch.sqrt(test).reshape(-1,resol_sc**2).to(device)+0j
    E_in[-1] = 0 
    E_in[-1, 80*40+40] = 1
    I_in = E_in.abs().cpu().reshape(-1,resol_sc,resol_sc)**2
    E_out = net_diff(E_in)
    I_out = E_out.abs().cpu().reshape(-1,resol_in,resol_in)**2

fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
digit = 0
ax[0,0].matshow(I_in[digit, 20:60,20:60].T, vmin=0, vmax=I_in[digit].max(),  cmap=plt.cm.inferno)
ax[1,0].matshow(I_out[digit].T, vmin=0, vmax=I_out[digit].max(),  cmap=plt.cm.inferno)
ax[0,1].matshow(I_in[-1, 20:60,20:60].T, vmin=0, )
ax[1,1].matshow(I_out[-1].T, vmin=0,)
fig.tight_layout()
print("Energy conserved: ", round((I_out[digit].sum()/I_in[digit].sum()*100).item(), 2), "%")
print("Energy conserved psf: ", round((I_out[-1].sum()/I_in[-1].sum()*100).item(), 2), "%")

print("I_in_max : ", I_in[digit].max())
print("I_out_max : ", I_out[digit].max())




#%%

test = test_set[:,0].abs()**2 #intensity
upsample = Upsample(size=(resol_out,resol_out), mode='bilinear')
test = upsample(test.reshape(test.shape[0],-1,*test.shape[1:]))

test_pad = torch.ones(10,1, resol_sc, resol_sc)
test_pad[:,:, int((resol_sc-resol_in)/2):int((resol_sc+resol_in)/2), int((resol_sc-resol_in)/2):int((resol_sc+resol_in)/2)] = test
test = test_pad.clone()





