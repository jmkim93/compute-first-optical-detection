#%% Dataset
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import matplotlib
import plot_setting
from scipy.optimize import curve_fit
from datetime import datetime

import torch
from  torch.nn.modules.upsampling import Upsample

torch.set_default_device("cuda" if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

import optics_module as om

import torchvision.datasets as ds
# from  torch.nn.modules.upsampling import Upsample

emnist_test = ds.EMNIST(root="emnist", split='digits', download=True, train=False)
torch.save(emnist_test.data, 'emnist_test_data.pt')
torch.save(emnist_test.targets, 'emnist_test_targets.pt')

# emnist_train = ds.EMNIST(root="emnist", split='digits', download=True, train=True)
# torch.save(emnist_train.data, 'emnist_train_data.pt')
# torch.save(emnist_train.targets, 'emnist_train_targets.pt')


N_class = 10
test_set = emnist_test.data[torch.argsort(emnist_test.targets)]+0.0
test_set = test_set.reshape(N_class, -1, 28, 28)[:,:1000]
del emnist_test

### Low contrast and unnormalized input - Fig2
Tmin = 300
Tmax = 310
ambient = om.photon_num_per_vol(10, torch.tensor(Tmin)).to('cpu')
test_set_intensity = om.photon_num_per_vol(10, Tmin+(Tmax-Tmin)*test_set/255)/ambient
torch.cuda.empty_cache()
# I_diff = (test_set_intensity.max()-test_set_intensity.min()).item()

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

test_set = dataset_field(test_set_intensity).cpu()
X_test = test_set.reshape(-1,resol_scene**2)
Y_test = (torch.arange(N_class).reshape(-1,1) * torch.ones(1,1000)).reshape(-1).cpu()

del test_set_intensity
torch.cuda.empty_cache()

resol_optics = [resol_scene, resol_out]
N_modes_digital = [resol_out**2, 300, 200, 50, 10]
p = 0.2

#%% Data calculation
train_optical = False
net = om.classifier_diffraction(resol_optics, W, W_lens, f, wvl, N_modes_digital, 1e5, 0, p,
                                incoherent=True, train=train_optical, batchnormfirst=True, affine=False, magnify=mag, Ndx=5)
checkpoint = torch.load("Data_Fig5/net_diff_quarternoise_f{}_train{}.pt".format(f,train_optical))
net.load_state_dict(checkpoint["net_state_dict_opt"])

with torch.no_grad():
    E_in = X_test.to(device)
    I_out = net[0](E_in).abs()**2
I_diff = (I_out.max()-I_out.min()).item()

dark_noise_level = np.linspace(0, I_diff, 21) 
N_repeat = 20

test_accuracy = []
test_pure_signal = []
test_noisy_signal = []

test_entropy = []

for train_optical in [False, True]:
    net = om.classifier_diffraction(resol_optics, W, W_lens, f, wvl, N_modes_digital, 1e5, 0, p,
                                    incoherent=True, train=train_optical, batchnormfirst=True, affine=False, magnify=mag, Ndx=5)
    checkpoint = torch.load("Data_Fig5/net_diff_quarternoise_f{}_train{}.pt".format(f,train_optical))
    net.load_state_dict(checkpoint["net_state_dict_opt"])

    for ii, I_dark in enumerate(dark_noise_level):
        print(ii)
        net[1].set(dt=1e6, dark=I_dark)
        net.eval()

        for repeat in range(N_repeat):
            with torch.no_grad():
                y = net(X_test.to(device))
                test_infer = torch.argmax(y, dim=1)
                acc = ((Y_test.to(device) == test_infer)+0.0).mean().item()
            test_accuracy.append(acc)

    net[1].set(dt=1e6, dark=I_diff/4)
    with torch.no_grad():
        E_out = net[0](X_test[np.arange(10)*1000].to(device))
        I_out = E_out.abs()**2
        I_noisy = net[1](E_out)
    test_pure_signal.append(I_out.cpu().numpy())
    test_noisy_signal.append(I_noisy.cpu().numpy())

    with torch.no_grad():
        I_out = net[0](X_test.to(device)).abs()**2
    I_min, I_max = I_out.min(dim=0)[0].reshape(28,28), I_out.max(dim=0)[0].reshape(28,28)
    np.savez("Data_Fig5/Iminmax_realIR_{}.npz".format(train_optical), 
             Imin=I_min.cpu().numpy(), Imax=I_max.cpu().numpy())


test_accuracy = np.array(test_accuracy).reshape(2, len(dark_noise_level), -1)

test_noisy_signal = np.array(test_noisy_signal).reshape(-1, *E_out.shape)
test_pure_signal = np.array(test_pure_signal).reshape(-1, *E_out.shape)

np.savez("Data_Fig5_diff/test.npz",
         noise=dark_noise_level,
         accuracy=test_accuracy, 
         noisysignal=test_noisy_signal,
         puresignal=test_pure_signal,
         lens1=net[0].lens1.cpu().numpy(),
         lens3=net[0].lens3.cpu().numpy(),
         phase1=net[0].phase1.cpu().detach().numpy(),
         phase2=net[0].phase2.cpu().detach().numpy(),
         phase3=net[0].phase3.cpu().detach().numpy(),
         Idiff=I_diff
)

#%% Data load

with np.load("Data_Fig5/test.npz") as plot_data:
    dark_noise_level = plot_data["noise"]
    test_accuracy = plot_data["accuracy"]
    test_noisy_signal = plot_data["noisysignal"]
    test_pure_signal = plot_data["puresignal"]
    lens1 = plot_data["lens1"]
    lens3 = plot_data["lens3"]
    phase1 = plot_data["phase1"]
    phase2 = plot_data["phase2"]
    phase3 = plot_data["phase3"]
    I_diff = plot_data["Idiff"]

x_lens = torch.arange(-W_lens/2, W_lens/2, dx).cpu()+dx/2
x_scene = torch.arange(-W/2, W/2, dx).cpu()+dx/2
x_image = torch.arange(-14*dx, 14*dx, dx).cpu()+dx/2
Xl, Yl = torch.meshgrid(x_lens.flip(dims=(0,)), x_lens)
Xs, Ys = torch.meshgrid(x_scene.flip(dims=(0,)), x_scene)
Xi, Yi = torch.meshgrid(x_image.flip(dims=(0,)), x_image)

Xl = Xl.numpy()
Yl = Yl.numpy()
Xs = Xs.numpy()
Ys = Ys.numpy()
Xi = Xi.numpy()
Yi = Yi.numpy()


# focal length regression
def convex(data, center_x, center_y, fx, fy, dphase):
    x = data[0]
    y = data[1]
    return -k0*((x-center_x)**2/(2*fx) + (y-center_y)**2/(2*fy)) + dphase

def concave(data, center_x, center_y, fx, fy, dphase):
    x = data[0]
    y = data[1]
    return k0*((x-center_x)**2/(2*fx) + (y-center_y)**2/(2*fy)) + dphase

X_data = Xl.reshape(-1)
Y_data = Yl.reshape(-1)
lens_data =  (phase1-lens1).reshape(-1)
XY_data = np.column_stack([X_data, Y_data]).T
parameters, covariances = curve_fit(convex, XY_data, lens_data)
center1 = parameters[:2]
focal_length1 = parameters[2:4]

lens_data =  (phase3-lens3).reshape(-1)
XY_data = np.column_stack([X_data, Y_data]).T
parameters, covariances = curve_fit(convex, XY_data, lens_data)
center3 = parameters[:2]
focal_length3 = parameters[2:4]

X_data = Xs.reshape(-1)
Y_data = Ys.reshape(-1)
lens_data =  (phase2).reshape(-1)
XY_data = np.column_stack([X_data, Y_data]).T
parameters, covariances = curve_fit(concave, XY_data, lens_data)
center2 = parameters[:2]
focal_length2 = parameters[2:4]


print(center1, focal_length1/f)
print(center2, focal_length2/f)
print(center3, focal_length3/f)

#%% Supplementary Fig, trained lens profile

digit = 0
vmin, vmax = test_pure_signal[0,digit].min(), test_pure_signal[:,digit].max()
fig, ax = plt.subplots(2,4, figsize=(6.5, 3.1))

pc00 = ax[0,0].pcolormesh(Xl,Yl, np.angle(np.exp(1j*lens1)).reshape(resol_lens,-1), vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight, linewidth=0,rasterized=True, shading='gouraud')
ax[0,0].set(xticks=(-50,0,50), yticks=(-50,0,50))
ax[0,2].pcolormesh(Xl,Yl, np.angle(np.exp(1j*lens3)).reshape(resol_lens,-1), vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight, linewidth=0,rasterized=True, shading='gouraud')
ax[0,2].set(xticks=(-50,0,50), yticks=(-50,0,50))
pc03 = ax[0,3].pcolormesh(Xi,Yi, test_pure_signal[0,digit].reshape(28,-1), vmin=vmin, vmax=vmax, cmap=plt.cm.inferno, linewidth=0,rasterized=True, shading='gouraud')
ax[0,3].set(xticks=(-20,0,20), yticks=(-20,0,20))


ax[1,0].pcolormesh(Xl,Yl, np.angle(np.exp(1j*(phase1-lens1))).reshape(resol_lens,-1), vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight, linewidth=0,rasterized=True, shading='gouraud')
ax[1,0].set(xticks=(-50,0,50), yticks=(-50,0,50))
ax[1,1].pcolormesh(Xs,Ys, np.angle(np.exp(1j*(phase2))).reshape(resol_scene,-1), vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight, linewidth=0,rasterized=True, shading='gouraud')
ax[1,1].set(xticks=(-40,-20,0,20,40), yticks=(-40,-20,0,20,40))
ax[1,2].pcolormesh(Xl,Yl, np.angle(np.exp(1j*(phase3-lens3))).reshape(resol_lens,-1), vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight, linewidth=0,rasterized=True, shading='gouraud')
ax[1,2].set(xticks=(-50,0,50), yticks=(-50,0,50))
pc13 = ax[1,3].pcolormesh(Xi,Yi, test_pure_signal[1,digit].reshape(28,-1), vmin=vmin, vmax=vmax, cmap=plt.cm.inferno, linewidth=0,rasterized=True, shading='gouraud')
ax[1,3].set(xticks=(-20,0,20), yticks=(-20,0,20))

[ax[i,j].axhline(y=0, color='w', lw=0.3, ls='--') for i in range(2) for j in (range(4) if i==1 else [0,2,3])]
[ax[i,j].axvline(x=0, color='w', lw=0.3, ls='--') for i in range(2) for j in (range(4) if i==1 else [0,2,3])]

c = "aqua"
ax[0,0].plot([-3,3],[0,0], c, lw=0.75)
ax[0,0].plot([0,0],[-3,3], c, lw=0.75)
ax[0,2].plot([-3,3],[0,0], c, lw=0.75)
ax[0,2].plot([0,0],[-3,3], c, lw=0.75)
ax[1,0].plot([center1[0]-focal_length1[0]/100, center1[0]+focal_length1[0]/100],[center1[1], center1[1]], c, lw=0.75)
ax[1,0].plot([center1[0], center1[0]],[center1[1]-focal_length1[1]/100, center1[1]+focal_length1[1]/100], c, lw=0.75)
ax[1,1].plot([center2[0]-focal_length2[0]/100, center2[0]+focal_length2[0]/100],[center2[1], center2[1]], c, lw=0.75)
ax[1,1].plot([center2[0], center2[0]],[center2[1]-focal_length2[1]/100, center2[1]+focal_length2[1]/100], c, lw=0.75)
ax[1,2].plot([center3[0]-focal_length3[0]/100, center3[0]+focal_length3[0]/100],[center3[1], center3[1]], c, lw=0.75)
ax[1,2].plot([center3[0], center3[0]],[center3[1]-focal_length3[1]/100, center3[1]+focal_length3[1]/100], c, lw=0.75)


ax[0,0].set_title(r"$z=f_0$", fontsize=7)
ax[0,1].set_title(r"$z=2f_0$", fontsize=7)
ax[0,2].set_title(r"$z=3f_0$", fontsize=7)
ax[0,3].set_title(r"Image, $z=4f_0$", fontsize=7)

ax[0,1].set(xticks=(), yticks=())
[ax[0,1].spines[pos].set_visible(False) for pos in ["top", "bottom", "left", "right"]]


W_im = 28*dx
tc = 'lime'
ax[0,0].text(-W_lens/2*0.9,W_lens/2*0.75, r'a', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')
ax[0,2].text(-W_lens/2*0.9,W_lens/2*0.75, r'b', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')
ax[0,3].text(-W_im/2*0.9,W_im/2*0.75, r'c', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')

ax[1,0].text(-W_lens/2*0.9,W_lens/2*0.75, r'd', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')
ax[1,1].text(-W/2*0.9,W/2*0.75, r'e', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')
ax[1,2].text(-W_lens/2*0.9,W_lens/2*0.75, r'f', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')
ax[1,3].text(-W_im/2*0.9,W_im/2*0.75, r'g', color=tc, fontweight='bold', fontsize=8, fontfamily='arial')



fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r"$x/\lambda$")
plt.ylabel(r"$y/\lambda$")


axins00 = ax[0,0].inset_axes([1.05, 0.25, 0.075, 0.5])
cb00 = plt.colorbar(pc00, orientation='vertical', cax=axins00, extend=None)
cb00.ax.set_yticks([0, np.pi])
cb00.ax.set_yticklabels([0, r"$\pi$"])

axins03 = ax[0,3].inset_axes([1.05, 0.25, 0.075, 0.5])
cb03 = plt.colorbar(pc03, orientation='vertical', cax=axins03, extend=None)
axins13 = ax[1,3].inset_axes([1.05, 0.25, 0.075, 0.5])
cb13 = plt.colorbar(pc13, orientation='vertical', cax=axins13, extend="min")


# cb03.ax.set_yticks([0.1, 0.2])
# cb03.ax.set_yticklabels([0, r"$\pi$"])
fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/SuppFig_lensprofile_{}.pdf".format(time), dpi=600, transparent=True)


#%% Fig. 5b

# fig, ax = plt.subplots(figsize=(2.7, 1.5))
# fig, ax = plt.subplots(figsize=(3.35, 1.6))
fig, ax = plt.subplots(figsize=(3, 1.5))
ax.plot(dark_noise_level/I_diff, test_accuracy.mean(axis=2)[0]*100, 'k', lw=0.8, label=r'$4f$ imaging')
ax.plot(dark_noise_level/I_diff, test_accuracy.mean(axis=2)[1]*100, color='r', ls='--', lw=0.8, label=r'Meta-imaging')

ax.plot(dark_noise_level[[0,5,10]]/I_diff, test_accuracy.mean(axis=2)[0,[0,5,10]]*100, 'o', color='k', ms=3, mew=0.5, mfc='white')
ax.plot(dark_noise_level[[0,5,10]]/I_diff, test_accuracy.mean(axis=2)[1,[0,5,10]]*100, 's', color='r', ms=3, mew=0.5, mfc='white')


ax.set(xlabel=r'Noise power, $\sigma_\mathrm{dark}/\Delta I$', xlim=(0,1),
       ylabel=r'Test accuracy (%)', ylim=(30, 100))
ax.legend(frameon=False, fontsize=7)
fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig5b_diff_{}.pdf".format(time), dpi=900, transparent=True)



#%% Fig. 5c

Wx= 28*10*dx
Wy= 28*2*dx

x = np.arange(-Wx/2,Wx/2, dx) + dx/2
y = np.arange(-Wy/2,Wy/2, dx) + dx/2
y = y[::-1]
X, Y = np.meshgrid(x,y)


pure_concat = np.transpose(test_pure_signal.reshape(2,10,28,28), axes=(0,3,1,2))[:,::-1,:,::-1].reshape(-1,10,28).reshape(56,-1)
noisy_concat = np.transpose(test_noisy_signal.reshape(2,10,28,28), axes=(0,3,1,2))[:,::-1,:,::-1].reshape(-1,10,28).reshape(56,-1)

fig, ax = plt.subplots(3,1, figsize=(3.35, 2.5), sharex=True,sharey=True)

vmin, vmax = test_pure_signal[0].min(), test_pure_signal.max()

I_dark = 0
gaussian = np.random.normal(loc=0, scale=I_dark, size=pure_concat.shape)
is0 = ax[0].pcolormesh(X, Y, pure_concat+gaussian, cmap=plt.cm.inferno, vmin=vmin,vmax=vmax, rasterized=True)

axins0 = ax[0].inset_axes([1.01, 0.1, 0.017, 0.8])
cb0 = plt.colorbar(is0, orientation='vertical', cax=axins0, extend='min')


I_dark = I_diff/4
gaussian = np.random.normal(loc=0, scale=I_dark, size=pure_concat.shape)
noisy_concat = pure_concat+gaussian
noisy_concat[:28] = noisy_concat[:28] - noisy_concat[:28].mean()
noisy_concat[28:] = noisy_concat[28:] - noisy_concat[28:].mean()
vmin, vmax = noisy_concat.min(),noisy_concat.max()
is1 = ax[1].pcolormesh(X,Y, noisy_concat, cmap=plt.cm.cividis, rasterized=True, vmin=vmin,vmax=vmax)

axins1 = ax[1].inset_axes([1.01, 0.1, 0.017, 0.8])
cb1 = plt.colorbar(is1, orientation='vertical', cax=axins1, extend=None)
cb1.ax.set_yticks([vmin,vmax])
cb1.ax.set_yticklabels(['min', 'max'])


I_dark = I_diff/2
gaussian = np.random.normal(loc=0, scale=I_dark, size=pure_concat.shape)
noisy_concat = pure_concat+gaussian
noisy_concat[:28] = noisy_concat[:28] - noisy_concat[:28].mean()
noisy_concat[28:] = noisy_concat[28:] - noisy_concat[28:].mean()
vmin, vmax = noisy_concat.min(),noisy_concat.max()
is2 = ax[2].pcolormesh(X,Y, noisy_concat, cmap=plt.cm.cividis, rasterized=True, vmin=vmin,vmax=vmax)
ax[2].plot([Wx*9/20,Wx*9/20+10], [Wy*0.4,Wy*0.4], 'aqua', lw=1)
# ax[2].text(Wx*8.5/20, Wy*0.2, r"$10\lambda$", fontsize=6, color='aqua')

ax[0].set(xlim=(-Wx/2,Wx/2), ylim=(-Wy/2,Wy/2), 
          xticks=np.arange(-Wx/2,Wx/2,Wx/10)+Wx/20, xticklabels=np.arange(10),
          yticks=[-Wy/4,Wy/4], yticklabels=['Meta', r'$4f$'])
ax[0].set_yticklabels(['Meta', r'$4f$'], rotation=60)
ax[1].set_yticklabels(['Meta', r'$4f$'], rotation=60)
ax[2].set_yticklabels(['Meta', r'$4f$'], rotation=60)

axins2 = ax[2].inset_axes([1.01, 0.1, 0.017, 0.8])
cb2 = plt.colorbar(is2, orientation='vertical', cax=axins2, extend=None)
cb2.ax.set_yticks([vmin,vmax])
cb2.ax.set_yticklabels(['min', 'max'])

[ax[i].axhline(y=0, color='w', lw=0.5, ls='-') for i in range(3)]

fig.tight_layout()
time = datetime.now().strftime("%Y%m%d%H%M")[2:]
fig.savefig("Figures/Fig5c_diff_{}.pdf".format(time), dpi=900, transparent=True)


#%% Fig. 5d Real IR image

from torchvision.transforms import Resize
from PIL import Image
image = np.asarray(Image.open('test_IR.jpg'))
image = image[:,640-512:640+512,0]

image = Resize(120)(torch.tensor(image).reshape(1,1024,-1))[0].numpy()
plt.matshow(image, cmap=plt.cm.gray)


