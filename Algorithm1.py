# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:18:25 2024

@author: COMEDIA
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from scipy import io as sio
from scipy.optimize import curve_fit

use_gpu = True
# Check GPU availability
if use_gpu:
    use_gpu = torch.cuda.is_available()
    device = "cuda:0"
else:
    device = "cpu"

#%% prepare the data

data_path = "data.mat" 
data_dict = sio.loadmat(data_path)
random_mask = data_dict['random_matrix']
imsave = data_dict['imsave']
refsave = data_dict['refsave']
chi_dimension = int(data_dict['chi_dimension'].squeeze())
numpics = imsave.shape[2]
imaging = np.sum(imsave[:,:,chi_dimension:-1],2)

#%%
hpos = 85
wpos = 85
neibor = 10 # data.mat is 3 dimension data (x,y,mask_i), pick one point in space to retrieval the chi^{(3)} 

data1 = (imsave[hpos-neibor:hpos+neibor, wpos-neibor:wpos+neibor,  :]).mean(axis=(0,1))
data1 = data1 - data1[-2]  
data1 = data1[1:-3]  
data1[data1 < 0] = 0  

data2 = refsave.mean(axis=(0,1))
data2 = data2 - data2[-2]  
data2 = data2[1:-3]

m = np.shape(random_mask)[0]
random_mask = random_mask[0:m,:]

random_mask = torch.Tensor(random_mask).to(device)
random_mask = random_mask + 1j*torch.zeros_like(random_mask)

n = chi_dimension
spectrum_RS = data1[:chi_dimension] 
spectrum_RS = torch.Tensor(spectrum_RS).to(device)
chi_amp = torch.sqrt(spectrum_RS)
chi_np = chi_amp.cpu().detach().numpy()
Y = data1[chi_dimension:]
Y = Y[0:m]
Y = torch.Tensor(Y).to(device)

ref_RS = data2[:chi_dimension]
E_broadband = np.sqrt(ref_RS)

# fitting E_broadband to a gaussian curve
x_data = np.linspace(0, len(E_broadband.squeeze()) - 1, len(E_broadband.squeeze()))
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))
params, covariance = curve_fit(gaussian, x_data, E_broadband.squeeze(), p0=[1, len(E_broadband.squeeze()) / 2, 10])
E_broadband_fit = gaussian(x_data, *params)
E_broadband = torch.Tensor(E_broadband_fit).to(device)
E_broadband = (E_broadband + 1j*torch.zeros_like(E_broadband)).squeeze()

#%% forward model 

def forward(chi, E_broadband, random_mask):

    m = random_mask.size()[0]
    n = random_mask.size()[1]
    
 
    E_as = F.conv1d( (E_broadband * random_mask * chi).unsqueeze(0), 
                     (E_broadband * random_mask).unsqueeze(1).flip(-1), 
                      padding=n-1,
                      groups=m # number of masks (m) different convolutions happening in parallel
                   ).squeeze()
    Y = torch.sum(torch.abs(E_as) ** 2, -1).squeeze()
    
    return Y

#%% reconstruct chi
# grad decend param set
n_iter = 20000

# Random initialization
chi_est = (torch.randn(n,) + 1j*torch.randn(n,)).to(device)
chi_est = chi_est/torch.max(torch.abs(chi_est))
chi_est.requires_grad = True
step =  4e-1

loss_vec = []
criterion = nn.MSELoss()
spec_grad_norms = []
image_grad_norms = []

# Params for a hacky way to reduce step size when cycles in loss occur
rearview = 10
max_inc = 4
step_reduce = 0.8
last_reduced_step = -np.inf

for i_iter in tqdm(range(n_iter)): 
   
    # Forward
    Y_est = forward(chi = chi_est, E_broadband = E_broadband,
                    random_mask = random_mask) 

    # Backprop loss
    loss = criterion(Y_est, Y)
    loss_vec.append(loss.item())
    loss.backward()
             
    # Update
    with torch.no_grad():
        if len(loss_vec) > rearview and np.sum(np.array(loss_vec[-rearview:]) - np.array(loss_vec[-rearview-1:-1]) > 0) > max_inc:
            if (i_iter - 2) > last_reduced_step: 
                step *= step_reduce
                print(f"new step size: {step:0.3e}")
                last_reduced_step = i_iter

        # Gradient Step
        chi_est  -= step * chi_est.grad/torch.norm(chi_est.grad)  
 
        # projection
        chi_est.data.imag = chi_est.imag.clip(min=0) # Send chi_spec_est.imag to [0, inf]
        

    # Reset gradients
    chi_est.grad.zero_()       

chi_est = chi_est/chi_est[0] *torch.abs(chi_est[0]) # using the first pixel(Non-resonant pixel) as ref phase

#%% show the result
if torch.sum(torch.imag(chi_est)) < 0:
    chi_est = torch.conj(chi_est)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
imag = torch.imag(chi_est.cpu().detach()).numpy()
real = torch.real(chi_est.cpu().detach()).numpy()
abs_val = torch.abs(chi_est.cpu().detach()).numpy()
ax.plot(imag, label='Imag', color='r', linewidth=2) 
ax.plot(real, label='Real', color='b', linewidth=2) 
ax.plot(abs_val, label='abs', color='black', linewidth=2) 
ax.set_ylabel('Amplitude', color='black',fontsize=14)
ax.legend(loc='upper left', fontsize=14) 
xticks = [0, 4, 8, 12, 16]
xticklabels = ['2800 cm$^{-1}$', '2850 cm$^{-1}$', '2900 cm$^{-1}$', '2950 cm$^{-1}$', '3000 cm$^{-1}$']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=14)
plt.tight_layout()
plt.show()