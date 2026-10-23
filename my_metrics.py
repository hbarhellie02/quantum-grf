import numpy as np
import metrics
import matplotlib.pyplot as plt

from scipy import ndimage as img
import cmocean


plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15

og_8by8 = "results/default-state-space-12-bits/values_8_8_chunks.csv"
pi_8by8 = (
    "results/modified-kernel/default-state-space-12-bits-sqrtpi/values_8_8_chunks.csv"
)
pixi_8by8 = "results/modified-kernel/default-state-space-12-bits-sqrtpi-xi/values_8_8_chunks.csv"

s3 = "results/seed/3/values_8_8_chunks.csv"
s42 = "results/seed/42/values_8_8_chunks.csv"
s64 = "results/seed/64/values_8_8_chunks.csv"
s1196 = "results/seed/1196/values_8_8_chunks.csv"
s1723 = "results/seed/1723/values_8_8_chunks.csv"
s4095 = "results/seed/4095/values_8_8_chunks.csv"


large_perlin = "results/perlin/perlin_32_256.csv"
classical_grf_dir = "results/classical-grf/oversample_5/data_240/"

og_data, og_grad = metrics.qgrf_stats(og_8by8, 8, 8)
pi_data, pi_grad = metrics.qgrf_stats(pi_8by8, 8, 8)
pixi_data, pixi_grad = metrics.qgrf_stats(pixi_8by8, 8, 8)

s3_d, s3_g = metrics.qgrf_stats(s3, 8, 8)
s42_d, s42_g = metrics.qgrf_stats(s42, 8, 8)
s64_d, s64_g = metrics.qgrf_stats(s64, 8, 8)
s1196_d, s1196_g = metrics.qgrf_stats(s1196, 8, 8)
s1723_d, s1723_g = metrics.qgrf_stats(s1723, 8, 8)
s4095_d, s4095_g = metrics.qgrf_stats(s4095, 8, 8)

perlin_data, perlin_grad = metrics.perlin_stats(large_perlin)
# perlin_amp = np.minimum(np.maximum(10 / 3 * perlin_data, -1), 1)
# perlin_amp = 2 / np.pi * np.arctan(15 * perlin_data)
perlin_amp = np.tanh(6 * perlin_data)
perlin_amp_grad = [img.sobel(perlin_amp, axis=0), img.sobel(perlin_amp, axis=1)]

clas_data, clas_grad_abs, clas_grad_angles = metrics.cgrf_stats(classical_grf_dir)
clas_data_d, clas_grad_abs_d, clas_grad_angles_d = metrics.cgrf_stats(
    classical_grf_dir, downsample=True
)

target_distr = metrics.target_value_stats(1, 16, 50000, 4, 1 / 8)


def multihist(ax, datas, labels, solids_mask, alpha=1):
    assert len(datas) == len(labels)
    assert len(labels) == len(solids_mask)
    for data, label, solid in zip(datas, labels, solids_mask):
        nbins = 1 + int(np.log2(len(data.flatten())))
        if solid:
            ax.hist(
                data.flatten(),
                density=True,
                bins=nbins,
                label=label,
                alpha=0.3,
            )
        else:
            ax.hist(
                data.flatten(),
                density=True,
                bins=nbins,
                label=label,
                histtype="step",
                alpha=alpha,
            )


def errorbarhist(ax, datas, label):
    flat = [d.flatten() for d in datas]
    nbins = int(np.log2(len(flat[0]))) + 1
    hists = [np.histogram(fl, bins=nbins, density=True) for fl in flat]
    just_vals = [hist[0] for hist in hists]
    stddevs = np.std(just_vals, axis=0)
    bin_edge = 0.5 * (hists[0][1][:-1] + hists[0][1][1:])
    ax.bar(
        bin_edge,
        np.mean(just_vals, axis=0),
        width=np.diff(hists[0][1]),
        yerr=stddevs,
        label=label,
        alpha=0.5,
        capsize=2.5,
        ec="black",
    )


plt.clf()
fig, ax = plt.subplots()
multihist(
    ax,
    [clas_data_d, og_data, s3_d, s42_d, s64_d, s1196_d, s1723_d, s4095_d],
    [
        "CGRF_d",
        "0",
        "3",
        "42",
        "64",
        "1196",
        "1723",
        "4095",
    ],
    [True, False, False, False, False, False, False, False],
)
plt.xlabel("value of the field")
plt.ylabel("probability density")
plt.legend()
plt.savefig("seeded_vals.png")


plt.clf()
fig, ax = plt.subplots()
multihist(
    ax,
    [clas_data, clas_data_d, target_distr, perlin_amp],
    ["class. GRF", "downsampled class. GRF", "direct sampling", "amplified Perlin"],
    [False, False, False, False],
    alpha=0.75,
)
errorbarhist(
    ax, [og_data, s3_d, s42_d, s64_d, s1196_d, s1723_d, s4095_d], "quantum GRF"
)
plt.xlabel("value of the field")
plt.ylabel("probability density")
plt.legend()
plt.savefig("errorbar_values.png")


normalization1 = 1 / 4 * 1 / (2 / 16)
normalization2 = 1 / 4 * 1 / (2 / 240)
perlin_amp_grad_abs = np.sqrt(perlin_amp_grad[0] ** 2 + perlin_amp_grad[1] ** 2)


s_gradients = [og_grad, s3_g, s42_g, s64_g, s1196_g, s1723_g, s4095_g]
normalization = (
    1 / 4 * 1 / (2 / 16)
)  # dividing by sum of weights in sobel filter and twice the lattice spacing from the central difference formula
s_abs = [clas_grad_abs_d * normalization] + [
    np.sqrt(g[0] ** 2 + g[1] ** 2) * normalization for g in s_gradients
]

plt.clf()
fig, ax = plt.subplots()
multihist(
    ax,
    [
        clas_grad_abs * normalization2,
        clas_grad_abs_d * normalization,
        perlin_amp_grad_abs * normalization,
    ],
    ["class. GRF", "downsampled class. GRF", "amplified Perlin"],
    [False, False, False],
    alpha=0.75,
)
ax.scatter(0, 0, s=0)
errorbarhist(
    ax,
    s_abs[1:],
    "quantum GRF",
)
plt.xlabel("magnitude of the field gradient")
plt.ylabel("probability density")
plt.legend()
plt.savefig("errorbar_grads.png")

# get angle distributions
plt.clf()
fig, ax = plt.subplots()
ticks = np.linspace(-1, 1, 5)
ax.set_xticks(np.pi * ticks, [rf"{i} $\pi$" for i in ticks])
perlin_amp_angles = np.arctan2(perlin_amp_grad[1], perlin_amp_grad[0])
multihist(
    ax,
    [clas_grad_angles, clas_grad_angles_d, perlin_amp_angles],
    ["CGRF", "CGRF_d", "ampl. Perlin"],
    [False, False, True],
)
plt.xlabel("direction of the field gradient")
plt.ylabel("probability density")
plt.legend()
plt.savefig("class_dirs.png")


plt.clf()
fig, ax = plt.subplots()
ticks = np.linspace(-1, 1, 5)
ax.set_xticks(np.pi * ticks, [rf"{i} $\pi$" for i in ticks])

s_gradients_even = [og_grad, s42_g, s64_g, s1196_g]
s_gradients_odd = [s3_g, s1723_g, s4095_g]
s_angles_even = [clas_grad_angles_d] + [
    np.arctan2(g[1], g[0]) for g in s_gradients_even
]
s_angles_odd = [clas_grad_angles_d] + [np.arctan2(g[1], g[0]) for g in s_gradients_odd]
multihist(
    ax,
    [s_angles_even[0]],
    [
        "downsampled class. GRF",
    ],
    [False],
)
errorbarhist(ax, s_angles_even[1:], "quantum GRF, even seeds")
plt.xlabel("direction of the field gradient")
plt.ylabel("probability density")
plt.legend(loc="lower left")
plt.savefig("seeded_dirs_even.png")

plt.clf()

fig, ax = plt.subplots()
ticks = np.linspace(-1, 1, 5)
ax.set_xticks(np.pi * ticks, [rf"{i} $\pi$" for i in ticks])

multihist(
    ax,
    [s_angles_odd[0]],
    [
        "CGRF_d",
    ],
    [False],
)
errorbarhist(ax, s_angles_odd[1:], "quantum GRF, odd seeds")
plt.xlabel("direction of the field gradient")
plt.ylabel("probability density")
plt.legend()
plt.savefig("seeded_dirs_odd.png")
