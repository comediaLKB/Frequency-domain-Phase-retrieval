# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:08:58 2024

@author: COMEDIA
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from scipy import io as sio

# In[2]:


def reset_gpu_mem(device):
    torch.cuda.reset_peak_memory_stats(device=device)

def get_max_gpu_mem(device):
    return torch.cuda.max_memory_allocated(device=device)

def print_max_gpu_mem(device):
    print(f"{get_max_gpu_mem(device)/1e9 :.2f}")
    

# In[5]:

use_gpu = True
# Check GPU availability
if use_gpu:
    use_gpu = torch.cuda.is_available()
    device = "cuda:0"
else:
    device = "cpu"

#%% read data

data_path = "data.mat"

data_dict = sio.loadmat(data_path)

random_mask = data_dict['random_matrix']
imsave = data_dict['imsave']
refsave = data_dict['refsave']
n_spec = int(data_dict['chi_dimension'].squeeze())

numpics = imsave.shape[2]
imsize = imsave.shape[0]

#%% prepare data
I = imsave - imsave[:,:,-2][:,:,np.newaxis] 
I = I[:,:,1:-3]
I[I<0] = 0
Y = I[:,:,n_spec:] 
Y = torch.Tensor(Y).to(device)
Y = Y.reshape(Y.size()[0]*Y.size()[1],Y.size()[2])

I_ref = refsave - refsave[:,:,-2][:,:,np.newaxis]
I_ref = I_ref[:,:,1:-3]
ref_RS = I_ref[:,:,:n_spec]

E_broadband = (np.sqrt(np.sqrt(ref_RS)))
E_broadband = torch.Tensor(E_broadband).to(device)

asdf = E_broadband

E_broadband = (E_broadband + 1j*torch.zeros_like(E_broadband)).squeeze()
E_broadband = E_broadband.reshape(E_broadband.size()[0]*E_broadband.size()[1], E_broadband.size()[2])

print(f"ref_RS \t\t{ref_RS.shape}")
print(f"E_broadband \t{E_broadband.shape}")

r = 3

m = 4 # how many mask we used

m_start = 21
Y = Y[:,m_start:m_start+m]
random_mask = random_mask[m_start:m_start+m,:]

imaging = np.sum(imsave[:,:,n_spec:n_spec+m],2)

fig, axs = plt.subplots(1, m, figsize=(15, 5))  
fig.suptitle("The raw measured images (corresponding to different DMD mask)", fontsize=16)
for i in range(m):
    axs[i].imshow(imsave[:, :, n_spec + m_start + 1 + i])
    axs[i].axis('off') 
    axs[i].set_title(f"Mask {i+1}", fontsize=10)  

plt.show()

random_mask = torch.Tensor(random_mask).to(device)
random_mask = random_mask + 1j*torch.zeros_like(random_mask)

print(f"random_mask \t{random_mask.shape}")
print(Y.shape)

#%% load the known spectrum

Y_est_species1 = torch.load('BSA_spectrum.pth')
Y_est_species2 = torch.load('pend_spectrum.pth')
Y_est_species3 = torch.load('NR_spectrum.pth')

fig, ax = plt.subplots(1,r, figsize=(10,5))

Y_est_species1_np = Y_est_species1.squeeze().detach().cpu().numpy()
Y_est_species2_np = Y_est_species2.squeeze().detach().cpu().numpy()
Y_est_species3_np = Y_est_species3.squeeze().detach().cpu().numpy()


ax[0].plot(np.abs(Y_est_species1_np), label='abs')
ax[0].plot(np.real(Y_est_species1_np), label='real')
ax[0].plot(np.imag(Y_est_species1_np), label='imag')
ax[0].legend()
ax[0].set_title("Species 1")


ax[1].plot(np.abs(Y_est_species2_np), label='abs')
ax[1].plot(np.real(Y_est_species2_np), label='real')
ax[1].plot(np.imag(Y_est_species2_np), label='imag')
ax[1].legend()
ax[1].set_title("Species 2")


ax[2].plot(np.abs(Y_est_species3_np), label='abs')
ax[2].plot(np.real(Y_est_species3_np), label='real')
ax[2].plot(np.imag(Y_est_species3_np), label='imag')
ax[2].legend()
ax[2].set_title("Species 3")

fig.suptitle("The input spectrum of Alg.2",fontsize=16)
plt.show()

# In[76]:

def forward(chi_spec, chi_image, E_pr):
    E_pr_chi_spec = E_pr * chi_spec.T.unsqueeze(1)
    
    n_spec = random_mask.size()[1]
    m = random_mask.size()[0]
    n_s = chi_image.size()[1]
    r = chi_spec.shape[1]
    numgroups = r*m

    Y = F.conv1d(
        E_pr_chi_spec.flatten(0,1).unsqueeze(0),
        E_pr.flatten(0,1).unsqueeze(1).flip(-1),# NOTE: need to flip to actually do convolution!
        padding=n_spec-1,
        groups=numgroups
    )

    # FIXME: check if use reshape or view
    Y = chi_image.T  @ Y.reshape(r, m*(2*n_spec-1) )
    Y = Y.reshape(n_s,m,2*n_spec-1)    
    Y = ((Y.abs())**2).sum(-1)

    return Y

#%% reconstruct chi
# ############################
# IMPORTANT: Set Rank(above)
# ############################

# grad descend param set
n_iter = 50000

step = 0.1e1
normed_grad_step = True
noisy_grad = True

if noisy_grad:
    noise_weight = 0.01

n_spec = random_mask.size()[1]
n_s = Y.size()[0]*Y.size()[1] 

# Initialization
chi_spec_est = torch.cat((Y_est_species1.unsqueeze(-1).detach().clone(), 
                          Y_est_species2.unsqueeze(-1).detach().clone(),
                          Y_est_species3.unsqueeze(-1).detach().clone()),
                         -1).to(device)

chi_image_est = torch.rand(r, imsize*imsize).to(device) 

chi_image_est.requires_grad = True

E_broadband_avg = E_broadband.mean(0) 
E_broadband_avg = E_broadband_avg.unsqueeze(0)
E_avg_pr = E_broadband_avg.unsqueeze(1).repeat(r,1,1) * random_mask.unsqueeze(0)

print(f"E_avg_pr \t{E_avg_pr.shape}") # pixels, masks, freqs
print(f"chi_spec_est \t{chi_spec_est.shape}") # pixels, masks, freqs
print(f"chi_image_est \t{chi_image_est.shape}") # pixels, masks, freqs

# gradient descent
loss_vec = []
criterion = nn.MSELoss()

spec_grad_norms = []
image_grad_norms = []

zeros_for_chi_image_est = 1j*torch.zeros_like(chi_image_est)

# Params for a hacky way to reduce step size when cycles in loss occur
rearview = 10
max_inc = 4
step_reduce = 0.8
last_reduced_step = -np.inf

# ##########################################
# Set the normalization Mode
# See brief description next to each for what they do
# ##########################################

for i_iter in tqdm(range(n_iter)): 
    # Forward

    Y_est = forward(chi_spec = chi_spec_est, 
                    chi_image = chi_image_est + zeros_for_chi_image_est, 
                    E_pr = E_avg_pr) #E_pr
    # Backprop loss
    loss = criterion(Y_est, Y)
    loss_vec.append(loss.item())
    loss.backward()
             
    # Update
    with torch.no_grad():
        if len(loss_vec) > rearview and np.sum(np.array(loss_vec[-rearview:]) - np.array(loss_vec[-rearview-1:-1]) > 0) > max_inc:
            if (i_iter - 2) > last_reduced_step: # FIXME: get rid of hard coding
                step *= step_reduce
                print(f"new step size: {step:0.3e}")
                last_reduced_step = i_iter

        # Gradient Step
        if noisy_grad:
            chi_image_est += noise_weight * step * torch.randn_like(chi_image_est)
            
        if normed_grad_step:
            chi_image_est -= step * chi_image_est.grad/torch.norm(chi_image_est.grad)
        else:
            chi_image_est -= step * chi_image_est.grad

        # Projection                 
        chi_image_est.data = chi_image_est.clip(min=0) 
        
    # Reset gradients       
    chi_image_est.grad.zero_() 

chi = torch.matmul(chi_spec_est, chi_image_est+1j*torch.zeros_like(chi_image_est)).T.unsqueeze(1)

vmin = min(chi_image_est[i].detach().cpu().min().item() for i in range(r))
vmax = max(chi_image_est[i].detach().cpu().max().item() for i in range(r))

fig, ax = plt.subplots(1,r, figsize=( r*2  ,3), dpi=150)

for i in range(r):
    ax[i].imshow(chi_image_est[i].reshape(imsize,imsize).detach().cpu().numpy(), cmap='gray',vmin=vmin, vmax=vmax)
    ax[i].set_title(f'EigImage {i}')
    
fig.suptitle('Reconstructed Eigen Images')