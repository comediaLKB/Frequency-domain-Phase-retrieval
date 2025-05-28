# Frequency-domain-Phase-retrieval
Python codes for frequency domain Phase retrieval applied to Coherent Anti-Stokes Raman Scattering

Corresponding results are presented in the paper "Computational field-resolved coherent chemical imaging"

Authors: Shupeng Zhao//Lea Chibani//Edward Chandler//Fangyu Liu//Jianqi Hu//Lorenzo Valzania//Ulugbek S. Kamilov//Hilton B. de Aguiar

contact: spzhao0724@gmail.com   //  version 11/2024


The function of "Algorithm1" and "Algorithm2" are described in the paper "Computational field-resolved coherent chemical imaging".

The file "data.mat" is a set of three-dimensional data, with the sample being BSA powder immersed in pentadecane liquid. The three dimensions correspond to different spatial points x,y and different masks on the DMD. The variable "refsave" save the local PDA measurement results of the pump beam. The variable "imsave" save the CARS signal measured by the PMT. The variable "random_mask" contains the mask loaded onto the DMD.  "data.mat" is the input for both "Algorithm1" and "Algorithm2".

"BSA_spectrum.pth","pend_spectrum.pth" and "NR_spectrum.pth" are the input of "Algorithm2".
