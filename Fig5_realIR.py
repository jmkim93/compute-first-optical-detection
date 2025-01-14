#%% Dataset
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import matplotlib
import plot_setting
from scipy.optimize import curve_fit
from datetime import datetime


from torchvision.utils import save_image
import torch
from  torch.nn.modules.upsampling import Upsample

torch.set_default_device("cuda" if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

import optics_module as om

import torchvision.datasets as ds


#%% Fig. 5d Real IR image
from torchvision.transforms import Resize
from PIL import Image


scene = torch.tensor(np.asarray(Image.open('test_IR.jpg')))
scene = scene[:,640-512:640+512,0]
# scene = Resize(1024)(torch.tensor(scene).reshape(1,1024,-1))[0]


with np.load("Data_Fig5/Iminmax_realIR_False.npz") as plot_data:
    I_min_4f = torch.tensor(plot_data["Imin"])
    I_max_4f = torch.tensor(plot_data["Imax"])
with np.load("Data_Fig5/Iminmax_realIR_True.npz") as plot_data:
    I_min_meta = torch.tensor(plot_data["Imin"])
    I_max_meta = torch.tensor(plot_data["Imax"])

minmax = Resize(1024)(torch.stack([I_min_4f,I_max_4f,I_min_meta,I_max_meta], dim=0))



I_diff = minmax[1].max()-minmax[0].min()
noise_power = I_diff/6
scene_4f = minmax[0] + (minmax[1]-minmax[0])* scene/256 + torch.normal(mean=0, std=torch.ones_like(scene)*noise_power)
scene_meta = minmax[2] + (minmax[3]-minmax[2])* scene/256 + torch.normal(mean=0, std=torch.ones_like(scene)*noise_power)

scene_4f = scene_4f.cpu()
scene_meta = scene_meta.cpu()

# scene_4f_noise = scene_4f - (minmax[0] + (minmax[1]-minmax[0])* scene/256)
# scene_meta_noise = scene_meta - (minmax[2] + (minmax[3]-minmax[2])* scene/256)
PSNR_4f = 20 * np.log10( (minmax[0] + (minmax[1]-minmax[0])* scene/256).max() / noise_power )
PSNR_meta = 20 * np.log10( (minmax[2] + (minmax[3]-minmax[2])* scene/256).max() / noise_power )
print(PSNR_4f, PSNR_meta)

fig, ax = plt.subplots(2,1)
ax[0].imshow(scene_4f,  cmap=plt.cm.gray)
ax[1].imshow(scene_meta, cmap=plt.cm.gray)


def make_img(temp, **kwargs):
    vmin = kwargs['vmin'] if 'vmin' in kwargs else temp.min()
    vmax = kwargs['vmax'] if 'vmax' in kwargs else temp.max()
    temp = torch.clip((temp-vmin)/(vmax-vmin), min=0, max=1)

    if 'cmap' in kwargs:
        rgb = torch.tensor(kwargs['cmap'](temp))[:,:,:3].permute((2,0,1))
    else:
        rgb = torch.stack([temp]*3, dim=0)
    return rgb


# rgb = make_img(scene)

save_image(make_img(scene, cmap=plt.cm.inferno), 'scene.jpg')
save_image(make_img(scene_4f, cmap=plt.cm.cividis),  'scene_4f.jpg')
save_image(make_img(scene_meta, cmap=plt.cm.cividis), 'scene_meta.jpg')
save_image(make_img(scene_4f[150:700,150:900]),  'scene_4f_crop.jpg')
save_image(make_img(scene_meta[150:700,150:900]), 'scene_meta_crop.jpg')

