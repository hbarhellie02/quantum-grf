# Copyright (C) 2025 Matthias Deiml, Daniel Peterseim - All rights reserved

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from util import save_plot


# Parameters for the experiment:
# Factor, by which the noise should be larger than the reference field
oversampling_ref = 5
# Number of quadrature points in each dimension for the reference field (1/h)
N_ref = 240
# Number of samples for estimating the error
samples = 100
# Number of quadrature points in each dimension for the approximations 1/(lh)
# Note that all these values are devisors of N_ref.
N_exp = np.array([10, 15, 20, 30, 40, 48, 60])
# Characteristic length of the covariance
xi = 0.075

# Some quantities derived from the above parameters:
# The radius r for each of the approximations
radius_exp = np.ceil(2 * np.pi * xi**2 * N_exp * N_exp) / N_exp * N_ref
print(f"Radius for approximations: {radius_exp * N_exp / N_ref}")
# The total size (in each dimension) of the noise generated for the reference
# field
noise_size = N_ref * (2 * oversampling_ref + 1)


def kernel(x):
    norm = np.linalg.norm(x, axis=-1)
    return 4 / (2 * xi**2 * np.pi) ** (2 / 4) * np.exp(-(norm**2) / xi**2)


# kernel_size = N_ref * (oversampling_ref + 1)
kernel_size = N_ref * (oversampling_ref)
coords_1d = np.arange(-kernel_size, kernel_size + 1) / N_ref

print("Precomputing kernels...")
# Here, we precompute the kernel for the reference field
x, y = np.meshgrid(coords_1d, coords_1d)
coords = np.stack((x, y), axis=-1)
kernel_ref = kernel(coords)

# Here, we precompute the coefficients needed to compute coarse noise variables
# for the approximations from the fine noise used for the reference.
coords_1d = np.arange(-noise_size, noise_size + 1) / N_ref
x, y = np.meshgrid(coords_1d, coords_1d)
coords = np.stack((x, y), axis=-1)
sinc_kernels = []
for n in N_exp:
    sinc_kernels.append(np.prod(np.sinc(coords * n), axis=-1))

# Array containing the total errors for each approximation degree
err_sum = np.zeros_like(N_exp, dtype=np.float64)

for j in range(samples):
    # We generate the fine noise and compute the reference random field
    noise = np.random.normal(size=(noise_size, noise_size))
    ref = convolve(noise, kernel_ref, mode="valid") / N_ref
    max = np.max(np.abs(ref))

    # We output one realization of a field by the sigmoid function (Figure 1),
    # one untransformed reference field (Figures 1 and 4(b)), and the noise used
    # for the reference (Figure 4(a)).
    if j == 0:
        print("Generating figures ...")
        print(f"max of reference: {4*max}")
        print(f"scaling of sigmoid: {2*max}")
        save_plot(-1 / (1 + np.exp(-4 * ref)), "sigmoid.png", max=2)
        save_plot(-ref, "sigmoid_untransformed.png", max=max)
        save_plot(ref, "moving_averages_reference.png", max=max * 1.1)
        save_plot(noise, "moving_averages_reference_noise.png")

    for i, n in enumerate(N_exp):
        # We convolve the fine noise with a sinc kernel ...
        sinc_convolved_noise = convolve(noise, sinc_kernels[i], mode="same")
        # ... and keep only the values corresponding to the coarse noise
        factor = N_ref // n
        radius = int(radius_exp[i])
        noise_exp = np.zeros_like(noise)
        start = oversampling_ref * N_ref - radius
        end = (oversampling_ref + 1) * N_ref + radius
        noise_exp[start:end:factor, start:end:factor] = sinc_convolved_noise[
            start:end:factor, start:end:factor
        ]

        # Note that noise_exp still has the same size as noise, but contains
        # mostly zeros. This means the quadrature can be computed the same
        # as for ref. Just now most of the summands will be zero.
        exp = convolve(noise_exp, kernel_ref, mode="valid") / N_ref

        if j == 0:
            # We output one realization of each approximation (Figure 4(c)) and
            # the corresponding noise (Figure 4(d))
            max_err = np.max(np.abs(ref - exp))
            save_plot(exp, f"moving_averages_{n}.png", max * 1.1)
            noise_exp = convolve(
                noise_exp[start:end, start:end], np.ones((factor, factor)), mode="full"
            )[:-factor, :-factor]
            save_plot(noise_exp, f"moving_average_{n}_noise.png")

            if i == len(N_exp) - 1:
                print("Estimating convergence ...")

        # Finally, we store the error of the approximation
        err_sum[i] += np.max(np.abs(ref - exp))
        if (j + 1) % 10 == 0 and i == len(N_exp) - 1:
            print(f"{j + 1} / {samples} done")

print(err_sum / samples)
plt.semilogy(radius_exp / N_ref * N_exp, err_sum / samples)
plt.show()
