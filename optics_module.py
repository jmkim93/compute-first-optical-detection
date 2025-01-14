#%%
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.parametrizations import orthogonal
import numpy as np
from tqdm import tqdm
from itertools import product


# Saturable absorber nonlinearity - not used
class sat_abs(nn.Module):
    def __init__(self, alphaz, I_sat, **kwargs):
        super(sat_abs, self).__init__()
        self.I_sat = I_sat
        self.alphaz = alphaz
        
    def forward(self, E_in):
        # I_in = E_in.abs()**2
        # phase_in = E_in.angle()
        # I_out = I_in * torch.exp(- self.alphaz /(1 + I_in/self.I_sat))
        # E_out = torch.sqrt(I_out) * torch.exp(1j*phase_in)
        E_out = E_in * torch.exp(-0.5*self.alphaz /(1 + E_in.abs()**2/self.I_sat))
        return E_out

# Saturable absorber relu nonlinearity - not used
class sat_abs_relu(nn.Module):
    def __init__(self, I_sat, **kwargs):
        super(sat_abs_relu, self).__init__()
        self.I_sat = I_sat
        
    def forward(self, E_in):
        I_in = E_in.abs()**2
        I_out = torch.nn.functional.relu(I_in-self.I_sat)
        # I_in = E_in.abs()**2
        phase_in = E_in.angle()
        # I_out = I_in * torch.exp(- self.alphaz /(1 + I_in/self.I_sat))
        E_out = torch.sqrt(I_out) * torch.exp(1j*phase_in)
        # E_out = E_in * torch.exp(-0.5*self.alphaz /(1 + E_in.abs()**2/self.I_sat))
        return E_out


# Detection module: input E, output |E|^2 + dark/photon noises
class detection(nn.Module):
    def __init__(self, **kwargs):
        super(detection, self).__init__()
        self.noise = kwargs['noise'] if 'noise' in kwargs else False
        self.dt = kwargs['dt'] if 'dt' in kwargs else 1e10
        self.I_dark = kwargs['dark'] if 'dark' in kwargs else 0
        # self.incoherent = kwargs["incoherent"] if "incoherent" in kwargs else False
        self.N_phase = kwargs["N_phase"] if "N_phase" in kwargs else 50

    def forward(self, E_in):
        I_out = E_in.abs()**2
        # if self.incoherent:
        #     I_out = I_out.reshape(self.N_phase, -1, I_out.shape[-1])
        #     I_out = I_out.mean(dim=0)
        if self.noise:
            I_out_detach = I_out.detach()
            poisson = torch.poisson(I_out_detach * self.dt) /self.dt - I_out_detach
            # poisson = torch.normal(torch.zeros_like(I_out_detach), torch.sqrt(I_out_detach/self.dt))
            gaussian = torch.normal(torch.zeros_like(I_out_detach), self.I_dark*torch.ones_like(I_out_detach))
            I_out = I_out + poisson + gaussian
            # I_out_detach = I_out.detach().cpu().numpy()
            # poisson = np.random.poisson(I_out_detach * self.dt) /self.dt - I_out_detach
            # gaussian = np.random.normal(np.zeros_like(I_out_detach), self.I_dark*np.ones_like(I_out_detach))
            # I_out = I_out + torch.tensor(poisson + gaussian).to(I_out.dtype)
        return I_out

    def set(self, **kwargs):
        self.dt = kwargs['dt'] if 'dt' in kwargs else self.dt
        self.I_dark = kwargs['dark'] if 'dark' in kwargs else self.I_dark


# class detection_incoherent(nn.Module):
#     def __init__(self, **kwargs):
#         super(detection_incoherent, self).__init__()
#         self.noise = kwargs['noise'] if 'noise' in kwargs else False
#         self.dt = kwargs['dt'] if 'dt' in kwargs else 1e10
#         self.I_dark = kwargs['dark'] if 'dark' in kwargs else 0

#     def forward(self, I_out):
#         I_out = torch.real(I_out)
#         if self.noise:
#             I_out_detach = I_out.detach()
#             poisson = torch.poisson(I_out_detach * self.dt) /self.dt - I_out_detach
#             gaussian = torch.normal(torch.zeros_like(I_out_detach), self.I_dark*torch.ones_like(I_out_detach))
#             I_out = I_out + poisson + gaussian
#         return I_out
    
#     def set(self, **kwargs):
#         self.dt = kwargs['dt'] if 'dt' in kwargs else self.dt
#         self.I_dark = kwargs['dark'] if 'dark' in kwargs else self.I_dark


# Identity operation (do nothing)
class identity(nn.Module):
    def __init__(self):
            super(identity, self).__init__()
    def forward(self, E_in):
        E_out = E_in
        return E_out
    

# Apply random phase to a field
class arbitrary_phase(nn.Module):
    def __init__(self, **kwargs):
            super(arbitrary_phase, self).__init__()
    def forward(self, E_in):
        E_out = E_in * torch.exp(2j * np.pi * torch.rand(E_in.shape))
        return E_out
    
# Apply random phase to a fied as a batch
class arbitrary_phase_multiple(nn.Module):
    def __init__(self, N_phase, **kwargs):
            super(arbitrary_phase_multiple, self).__init__()
            self.N_phase = N_phase
    def forward(self, E_in):
        E_out = E_in * torch.exp(2j * np.pi * torch.rand(self.N_phase, *E_in.shape))
        return E_out.reshape(-1, E_in.shape[-1])

# After detection, select a few detection values given by indices
class detection_filter(nn.Module):
    def __init__(self, indices):
            super(detection_filter, self).__init__()
            self.indices = indices
    def forward(self, I_in):
        I_out = I_in[:, self.indices]
        return I_out

# Apply max-pooling after detectors. 
class maxpool_layer(nn.Module):
    def __init__(self, N_seg):
            super(maxpool_layer, self).__init__()
            self.N_seg = N_seg
            self.kernel = int(28/N_seg)
            self.pooling = nn.MaxPool2d(self.kernel, stride=self.kernel)
    def forward(self, I_in):
        I_in = I_in.reshape(-1, 28, 28)
        I_out = self.pooling(I_in).reshape(-1, self.N_seg**2)
        return I_out


# pruning: replace detection value with a given input
class prune(nn.Module):
    def __init__(self, mask, **kwargs):
        super(prune, self).__init__()
        self.mask = mask
        self.value = kwargs['value'] if 'value' in kwargs else 0
  
    def forward(self, I):
        I[:,self.mask]=self.value.reshape(1,-1)
        return I
 
# Not used 
class orthogonal_Linear(nn.Module):
    def __init__(self, N):
        super(orthogonal_Linear, self).__init__()
        self.weight = nn.Parameter(torch.rand(N, N)+0j)

    def forward(self, E_in):
        skew_Hermitian = 0.5*(self.weight - torch.conj(self.weight.T))
        unitary = torch.matrix_exp(skew_Hermitian)
        E_out = E_in @ unitary
        return E_out
    

# For incoherent light, calculate I_out = SI_in instead of E_out = PE_in
class intensity_linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(intensity_linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, E_in):
        I_in = E_in.abs()**2
        weight = self.linear.weight.abs()**2
        weight = weight/weight.sum(dim=0, keepdim=True)
        I_out = I_in @ weight.T
        E_out = 0j + I_out**0.5
        return E_out
    
    def get_weight(self):
        weight = self.linear.weight.abs()**2
        weight = weight/weight.sum(dim=0, keepdim=True)
        return weight.detach()


# Generate free-space propagation kernel
def prop_matrix(resol_in, resol_out, dx, H, wvl, Ndx=1):
    W_in = resol_in * dx
    W_out = resol_out * dx
    x_in = (torch.arange(-W_in/2, W_in/2, dx) + dx/2).reshape(1,1,resol_in,1)
    y_in = (torch.arange(-W_in/2, W_in/2, dx) + dx/2).reshape(1,1,1,resol_in)
    x_out = (torch.arange(-W_out/2, W_out/2, dx) + dx/2).reshape(resol_out,1,1,1)
    y_out = (torch.arange(-W_out/2, W_out/2, dx) + dx/2).reshape(1,resol_out,1,1)
    
    shift = np.arange(-dx/2, dx/2, dx/Ndx) + dx/(2*Ndx)
    k0 = 2*np.pi/wvl
    prop = 0
    for del_x, del_y in product(shift, repeat=2):
        r = torch.sqrt((x_in-x_out+del_x)**2 + (y_in-y_out+del_y)**2 + H**2)
        prop += (1/wvl) * (1/(k0*r)-1j) * (H/r**2)  * torch.exp(1j*k0*r)
    prop = prop * (dx/Ndx)**2
    return prop.reshape(-1, resol_in,resol_in).reshape(-1, resol_in**2)


# OPU as a sequence of linear/nonlinear/... 
def optical_net(N_modes, I_sat, **kwargs):
    real = kwargs["real"] if "real" in kwargs else False
    incoherent = kwargs["incoherent"] if "incoherent" in kwargs else False
    sequential = []
    # assume real-valued weight matrix. not really used for this study.
    if real:
        for ii in range(len(N_modes)-2):
            sequential.append(orthogonal(nn.Linear(N_modes[ii], N_modes[ii+1], bias=False)))
            sequential.append(sat_abs_relu(I_sat))
        sequential.append(orthogonal(nn.Linear(N_modes[-2], N_modes[-1], bias=False)))
    # assume complex-valued weight matrix.
    else:
        for ii in range(len(N_modes)-2):
            sequential.append(
                intensity_linear(N_modes[ii], N_modes[ii+1]) if incoherent else 
                orthogonal(nn.Linear(N_modes[ii], N_modes[ii+1], bias=False).to(torch.cfloat))
            )
            sequential.append(sat_abs_relu(I_sat))
        sequential.append(
            intensity_linear(N_modes[-2], N_modes[-1]) if incoherent else 
            orthogonal(nn.Linear(N_modes[-2], N_modes[-1], bias=False).to(torch.cfloat))
        )
    return nn.Sequential(*sequential)


# Digital postprocessing by deep learning - not used
def digital_net(N_modes, p):
    sequential = []
    for ii in range(len(N_modes)-2):
        sequential.append(nn.Dropout1d(p=p))
        sequential.append(nn.Linear(N_modes[ii], N_modes[ii+1]))
        sequential.append(nn.BatchNorm1d(N_modes[ii+1]))
        sequential.append(nn.GELU())
    sequential.append(nn.Dropout1d(p=p))
    sequential.append(nn.Linear(N_modes[-2], N_modes[-1]))
    sequential.append(nn.BatchNorm1d(N_modes[-1]))
    sequential.append(nn.Sigmoid())
    return nn.Sequential(*sequential)

# Digital postprocessing by deep learning with logit output
def digital_net_logit(N_modes, p, **kwargs):
    batchnormfirst = kwargs["batchnormfirst"] if "batchnormfirst" in kwargs else False
    affine = kwargs["affine"] if "affine" in kwargs else True
    sequential = []
    # Normalize the detection value before passing to post-processing for stability.
    if batchnormfirst:
        sequential.append(nn.BatchNorm1d(N_modes[0], affine=affine))
    for ii in range(len(N_modes)-2):
        sequential.append(nn.Dropout1d(p=p))
        sequential.append(nn.Linear(N_modes[ii], N_modes[ii+1]))
        sequential.append(nn.BatchNorm1d(N_modes[ii+1], affine=affine))
        sequential.append(nn.GELU())
    sequential.append(nn.Dropout1d(p=p))
    sequential.append(nn.Linear(N_modes[-2], N_modes[-1]))
    if not batchnormfirst:
        sequential.append(nn.BatchNorm1d(N_modes[-1], affine=affine))
    return nn.Sequential(*sequential)

# Not used
def autoencoder(N_modes, alphaz, I_sat, dt, Idark, p):
    sequential = [
        optical_net(N_modes, alphaz, I_sat),
        detection(noise=True, dt=dt, dark=Idark),
        digital_net(N_modes[::-1], p)
    ]
    return nn.Sequential(*sequential)

# Not used
def autoencoder_prune(autoencoder, mask, value):
    sequential = [
        autoencoder[0],
        autoencoder[1],
        prune(mask, value=value),
        autoencoder[2]
    ]
    return nn.Sequential(*sequential)

# OPU-Detection-Post processing combined classifer model
def classifier(N_modes_optics, N_modes_digital, I_sat, dt, Idark, p, **kwargs):
    optical = kwargs["optical"] if "optical" in kwargs else True
    digital = kwargs["digital"] if "digital" in kwargs else True
    real = kwargs["real"] if "real" in kwargs else False
    bnf = kwargs["batchnormfirst"] if "batchnormfirst" in kwargs else False
    affine = kwargs["affine"] if "affine" in kwargs else True
    maxpool = kwargs["maxpool"] if "maxpool" in kwargs else 28
    # incoherent = kwargs["incoherent"] if "incoherent" in kwargs else (maxpool>1)
    incoherent = kwargs["incoherent"] if "incoherent" in kwargs else False
    # N_phase = kwargs["N_phase"] if "N_phase" in kwargs else 100
    sequential = [
        # optical_net(N_modes_optics, I_sat, real=real, incoherent=incoherent, N_phase=N_phase) if optical else identity(),
        # detection(noise=True, dt=dt, dark=Idark, incoherent=incoherent, N_phase=N_phase),
        optical_net(N_modes_optics, I_sat, real=real, incoherent=incoherent) if optical else identity(),
        detection(noise=True, dt=dt, dark=Idark),
        maxpool_layer(maxpool) if maxpool!=28 else (detection_filter(kwargs["filter"]) if "filter" in kwargs else identity()),
        digital_net_logit(N_modes_digital, p, batchnormfirst=bnf, affine=affine) if digital else identity()
    ]
    return nn.Sequential(*sequential)


# (Fig 5) 4f and Metaimaging system with diffractive optics
class optical_diffraction(nn.Module):
    def __init__(self, dim1d_inout, W, W_lens, f, wvl, **kwargs):
        super(optical_diffraction, self).__init__()
        self.wvl = wvl
        self.k0 = 2*np.pi/wvl
        self.W = W
        self.W_lens = W_lens
        self.f = f
        self.resol = dim1d_inout[0]
        self.dim_out = dim1d_inout[1]
        self.resol_lens = int(self.resol * W_lens/W)
        self.dx = W/self.resol
        self.incoherent = kwargs["incoherent"] if "incoherent" in kwargs else False
        self.device = kwargs["device"] if "device" in kwargs else "cpu"
        self.Ndx = kwargs["Ndx"] if "Ndx" in kwargs else 1
        self.mag = kwargs["magnify"] if "magnify" in kwargs else 1
        self.train_optical = kwargs["train"] if "train" in kwargs else False
        
        # self.H1, self.H2 = [(self.mag+1)/self.mag * f, (self.mag+1) * f] 
        # self.prop1 = prop_matrix(self.resol, self.resol_lens, self.dx, self.H1, wvl, self.Ndx)
        # self.prop2 = prop_matrix(self.resol_lens, self.resol,  self.dx, self.H2, wvl, self.Ndx)
        # xl = (torch.arange(-W_lens/2, W_lens/2, self.dx) + self.dx/2).reshape(self.resol_lens,1)
        # yl = (torch.arange(-W_lens/2, W_lens/2, self.dx) + self.dx/2).reshape(1,self.resol_lens)
        # self.lens = (self.k0 * (W_lens**2/2 - xl**2 - yl**2)/(2*f)).reshape(-1)
        # self.phase = nn.Parameter(self.lens.clone(), requires_grad=True) if self.train_optical else self.lens.clone()
        
        self.H1, self.H2, self.H3, self.H4 = [f, f, f*self.mag, f*self.mag] 
        self.prop1 = prop_matrix(self.resol, self.resol_lens, self.dx, self.H1, wvl, self.Ndx)
        self.prop2 = prop_matrix(self.resol_lens, self.resol,  self.dx, self.H2, wvl, self.Ndx)
        self.prop3 = prop_matrix(self.resol, self.resol_lens, self.dx, self.H3, wvl, self.Ndx)
        self.prop4 = prop_matrix(self.resol_lens, self.resol,  self.dx, self.H4, wvl, self.Ndx)
        xl = (torch.arange(-W_lens/2, W_lens/2, self.dx) + self.dx/2).reshape(self.resol_lens,1)
        yl = (torch.arange(-W_lens/2, W_lens/2, self.dx) + self.dx/2).reshape(1,self.resol_lens)
        
        self.lens1 = (self.k0 * (W_lens**2/2 - xl**2 - yl**2)/(2*f)).reshape(-1)
        self.lens3 = (self.k0 * (W_lens**2/2 - xl**2 - yl**2)/(2*f*self.mag)).reshape(-1)
    
        self.phase1 = nn.Parameter(self.lens1.clone(), requires_grad=True) if self.train_optical else self.lens1.clone()
        self.phase2 = nn.Parameter(torch.zeros(self.resol**2), requires_grad=True) if self.train_optical else torch.zeros(self.resol**2)
        self.phase3 = nn.Parameter(self.lens3.clone(), requires_grad=True) if self.train_optical else self.lens3.clone()

    # def amp_activation(self):
    #     return nn.functional.sigmoid(10*(self.amp - 0.5))

    def forward(self, E_in):
        # transfer = self.prop2 @ torch.diag(torch.exp(1j*self.phase2)) @ self.prop1 @ torch.diag(torch.exp(1j*self.phase1))
        transfer = self.prop4 @ torch.diag(torch.exp(1j*self.phase3)) @ self.prop3 @ torch.diag(torch.exp(1j*self.phase2)) @ self.prop2 @ torch.diag(torch.exp(1j*self.phase1))  @ self.prop1 
        if self.incoherent:
            I_in = E_in.abs()**2
            I_out = I_in @ (transfer.abs()**2).T
            E_out = 0j + I_out**0.5
        else:
            E_out = E_in @ (transfer.T)
        
        if self.dim_out != self.resol:
            a, b = int((self.resol-self.dim_out)/2), int((self.resol+self.dim_out)/2)
            E_out = E_out.reshape(-1, self.resol, self.resol)[:, a:b, a:b].reshape(-1, self.dim_out**2)
        return E_out

# (Fig 5) Classifier with diffractive optics
def classifier_diffraction(dim1d_inout, W, W_lens, f, wvl, N_modes_digital, dt, Idark, p, **kwargs):
    digital = kwargs["digital"] if "digital" in kwargs else True
    bnf = kwargs["batchnormfirst"] if "batchnormfirst" in kwargs else False
    affine = kwargs["affine"] if "affine" in kwargs else True
    incoherent = kwargs["incoherent"] if "incoherent" in kwargs else False
    train_optical = kwargs["train"] if "train" in kwargs else False

    mag = kwargs["magnify"] if "magnify" in kwargs else 1
    Ndx = kwargs["Ndx"] if "Ndx" in kwargs else 1
    sequential = [
        optical_diffraction(dim1d_inout, W, W_lens, f, wvl, 
                            incoherent=incoherent, magnify=mag, Ndx=Ndx, train=train_optical),
        detection(noise=True, dt=dt, dark=Idark),
        identity(),
        digital_net_logit(N_modes_digital, p, batchnormfirst=bnf, affine=affine) if digital else identity()
    ]
    return nn.Sequential(*sequential)


# def classifier_incoherent(N_modes_optics, N_modes_digital, I_sat, dt, Idark, p, **kwargs):
#     optical = kwargs["optical"] if "optical" in kwargs else True
#     digital = kwargs["digital"] if "digital" in kwargs else True
#     bnf = kwargs["batchnormfirst"] if "batchnormfirst" in kwargs else False
#     affine = kwargs["affine"] if "affine" in kwargs else True
#     maxpool = kwargs["maxpool"] if "maxpool" in kwargs else 28
#     sequential = [
#         Linear_incoherent(N_modes_optics[0],N_modes_optics[1]) if optical else identity(),
#         detection_incoherent(noise=True, dt=dt, dark=Idark),
#         maxpool_layer(maxpool) if maxpool!=28 else (detection_filter(kwargs["filter"]) if "filter" in kwargs else identity()),
#         digital_net_logit(N_modes_digital, p, batchnormfirst=bnf, affine=affine) if digital else identity()
#     ]
#     return nn.Sequential(*sequential)


# (Fig 4) Classifier with pruning included
def classifier_prune(classifier, mask, value):
    sequential = [
        classifier[0],
        classifier[1],
        prune(mask, value=value),
        classifier[-1]
    ]
    return nn.Sequential(*sequential)



# Train a model
def run(net, optimizer, criterion, N_epoch, loader, loader_val, device, **kwargs):
    
    resume = kwargs["resume"] if "resume" in kwargs else False
    real = kwargs["real"] if "real" in kwargs else False
    randshift = kwargs["randshift"] if "randshift" in kwargs else False

    incoherent = kwargs["incoherent"] if "incoherent" in kwargs else False
    nonunitary = kwargs["nonunitary"] if "nonunitary" in kwargs else False

    record = kwargs["tensorboard"] if "tensorboard" in kwargs else False
    if record:
        writer = kwargs["writer"] if "writer" in kwargs else SummaryWriter()

    if not resume:
        train_loss = []
        val_loss = []
        opt_state_dict = net.state_dict().copy()
        Epoch_start = 0
    else:
        train_loss = [tl for tl in kwargs["trainloss"]]
        val_loss = [vl for vl in kwargs["valloss"]]
        opt_state_dict = kwargs["optstatedict"]
        Epoch_start = len(kwargs["trainloss"])   

    for epoch in tqdm(range(Epoch_start, Epoch_start+N_epoch)):
        net.train()
        trls = 0
        counter = 0
        for E_in, target in loader:
            E_in, target = E_in.to(device), target.to(device)

            if (not incoherent) or nonunitary:
                    y = net(E_in if not real else torch.real(E_in))
            else:
                I_in = E_in.abs()**2
                W = (net[0][0].weight.abs()**2).T
                I_out = I_in @ W 
                I_out_detach = I_out.detach()
                poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
                gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
                x = I_out + poisson + gaussian
                y = net[3](net[2](x))

            loss = criterion(y, target)
            trls += loss.item()
            loss.backward()
            optimizer.step()
            counter += 1
            torch.cuda.empty_cache()
        train_loss.append(trls/counter)
        if record:
            writer.add_scalar("Loss/train", trls/counter, epoch)

        net.eval()
        valls = 0
        counter = 0
        with torch.no_grad():
            for E_in, target in loader_val:
                counter += 1
                E_in, target = E_in.to(device), target.to(device)
                if (not incoherent) or nonunitary:
                    y = net(E_in if not real else torch.real(E_in))
                else:
                    I_in = E_in.abs()**2
                    W = (net[0][0].weight.abs()**2).T
                    I_out = I_in @ W 
                    I_out_detach = I_out.detach()
                    poisson = torch.poisson(I_out_detach * net[1].dt) /net[1].dt - I_out_detach
                    gaussian = torch.normal(torch.zeros_like(I_out_detach), net[1].I_dark*torch.ones_like(I_out_detach))
                    x = I_out + poisson + gaussian
                    y = net[3](net[2](x))
                    
                loss = criterion(y, target)
                valls += loss.item()
            val_loss.append(valls/counter)
            if record:
                writer.add_scalar("Loss/val", valls/counter, epoch)

        if valls/counter < min(val_loss):
            min_val_loss = valls/counter
            opt_state_dict = net.state_dict().copy()

        torch.cuda.empty_cache()

    return np.array(train_loss), np.array(val_loss), opt_state_dict




### Unitary Matrix definitions

# Random (Haar) unitary matrix
def HaarRandU(**kwargs):
    N = kwargs["N"] if "N" in kwargs else 28
    sd = kwargs["seed"] if "seed" in kwargs else np.random.randint(0, 1e10)
    torch.manual_seed(sd)
    A = torch.normal(torch.zeros(2, N**2,N**2), torch.ones(2, N**2,N**2))
    A = A[0] + 1j* A[1]
    Q, R = torch.linalg.qr(A)
    Lambda = torch.diag( torch.diag(R) / torch.diag(R).abs() )
    return Q@Lambda


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

def block_focusing_intensity(n_list, fr=1):
    N = sum(n_list)
    starting = np.concatenate([np.array([0]), np.cumsum(n_list)[0:-1]])
    ending = np.cumsum(n_list)
    k = torch.arange(N).reshape(-1,1,1,1)
    l = torch.arange(N).reshape(1,-1,1,1)
    m = torch.arange(N).reshape(1,1,-1,1)
    n = torch.arange(N).reshape(1,1,1,-1)
    F_focus = [fr*(k==x0)*(l==y0)*(m>=x0)*(m<x1)*(n>=y0)*(n<y1) for x0, x1 in zip(starting, ending) for y0, y1 in zip(starting, ending)]
    F_outfocus = [(1-fr)/((x1-x0)*(y1-y0)-1)*((k>x0)*(l>=y0)+(k>=x0)*(l>y0))*(k<x1)*(l<y1)*(m>=x0)*(m<x1)*(n>=y0)*(n<y1) for x0, x1 in zip(starting, ending) for y0, y1 in zip(starting, ending)]
    F = torch.stack(F_focus+F_outfocus, dim=0).sum(dim=0).reshape(-1, N, N).reshape(-1, N**2)+1e-10
    filt = np.sort([a*N+b for a in starting for b in starting])
    return F, filt




### Physics: Blackbody radiation related functions

k_B = 8.617333262e-5 # [eV/K]
h = 4.135667696e-15 # [eV*s]
hbar = h/(2*np.pi)
c = 299792458e6 #[um/s] 
q_0 = 1.60217663e-19 # [coulombs]
occupation_num = lambda lda, T: 1/(torch.exp(h*c/(lda*k_B * T)) - 1)
std_num = lambda lda, T: torch.sqrt(occupation_num(lda,T)*(1+occupation_num(lda,T)))
DOS_per_vol = lambda lda: 8*np.pi/(lda**4)
photon_num_per_vol = lambda lda, T: occupation_num(lda, T)*DOS_per_vol(lda)
photon_flux_per_area = lambda lda, T: photon_num_per_vol(lda, T) * (c/4)
energy_spectral_per_vol = lambda lda, T: photon_num_per_vol(lda, T) * (h*c/lda)
