# Copyright (C) 2025 Matthias Deiml, Daniel Peterseim - All rights reserved

import numpy as np
from scipy.signal import convolve
from util import save_plot

# This computes a refrence value for the quantity of interest E[Z_left Z_right]
# as defined in Section 6.3.
# The code is largely duplicate to moving_averages.py

# Parameters for the experiment:
# Factor, by which the noise should be larger than the reference field
oversampling_ref = 5
# Number of quadrature points in each dimension for the reference field (1/h)
N_ref = 240
# Number of samples
samples = 10000
# Characteristic length of the covariance
xi = np.sqrt(1/8)

# The total size (in each dimension) of the noise generated for the reference
# field
noise_size = N_ref * (2 * oversampling_ref + 1)

def kernel(x):
    norm = np.linalg.norm(x, axis=-1)
    return 4 / (2 * xi**2 * np.pi) ** (2 / 4) * np.exp(-(norm**2) / xi**2)


print("Precomputing kernel...")
kernel_size = N_ref * (oversampling_ref)
coords_1d = np.arange(-kernel_size, kernel_size + 1) / N_ref
x, y = np.meshgrid(coords_1d, coords_1d)
coords = np.stack((x, y), axis=-1)
kernel_ref = kernel(coords)

sum = 0

for j in range(samples):
    noise = np.random.normal(size=(noise_size, noise_size))
    ref = np.cos(convolve(noise, kernel_ref, mode="valid") / N_ref)
    if j == 0:
        print("Generating figure ...")
        save_plot(ref, "quantum_circuit_cos_reference.png")
    left = np.average(ref[N_ref // 2 :, :])
    right = np.average(ref[: N_ref // 2, :])
    cov = left * right
    sum += cov
    print(sum/(j + 1))

print(sum / samples)
