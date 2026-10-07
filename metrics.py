import qiskit as qk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
import cmocean
from qiskit_aer import AerSimulator
from scipy import ndimage as img
import os

from random_generator import pcg6to4, pcg_k_to_l
from util import save_plot
from quantum_circuit import gaussian_random_field, chunked_gaussian_random_field
from user_interface import (
    run_chunk_circuit,
    run_default_circuit,
    parse_chunk,
    parse_perlin_csv,
)

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeAachen


def gaussian_covariance(x, y, C, xisr):
    return C * np.exp(-(np.abs(x - y) ** 2) / (2 * xisr))


def gaussian_cov_kernel(x, y, C, xisr):
    return np.sqrt(C) / np.sqrt(xisr * np.pi / 2) * np.exp(-(np.abs(x - y) ** 2) / xisr)


def basic_quantum_stats():
    pcg6 = pcg6to4(3, 17)
    grf = gaussian_random_field(pcg6)

    backend = AerSimulator()

    grf_transpiled = qk.transpile(grf, backend)

    print("\nDefault GRF, on ideal system")

    print(f"operations: {grf_transpiled.count_ops()}")

    print(f"full depth: {grf_transpiled.depth()}")
    print(
        f"nonlocal gate depth: {grf_transpiled.depth(lambda instr: len(instr.qubits) > 1)}"
    )
    # print(f"approx. runtime: {grf_transpiled.estimate_duration(backend.target)}")

    print("\n --------------------------------- \n")

    backend = FakeAachen()

    grf_transpiled = qk.transpile(grf, backend)

    print("\nDefault GRF, on simulated IBM Heron r3 (Aachen)")

    print(f"operations: {grf_transpiled.count_ops()}")

    print(f"full depth: {grf_transpiled.depth()}")
    print(
        f"nonlocal gate depth: {grf_transpiled.depth(lambda instr: len(instr.qubits) > 1)}"
    )
    print(f"approx. runtime: {grf_transpiled.estimate_duration(backend.target)}")

    print("\n --------------------------------- \n")


def qgrf_stats(file, chunks_x, chunks_y, save=False):
    """
    calculates sobel-filter based gradient for a quantum GRF  file
    if `save` is set, it will save histograms of the value and gradient distributions
    otherwise, the arrays of values and full gradient will be returned
    """
    data = np.zeros((16 * chunks_x, 16 * chunks_y))
    for i in range(chunks_x):
        for j in range(chunks_y):
            data[16 * i : 16 * (i + 1), 16 * j : 16 * (j + 1)] = np.reshape(
                parse_chunk(file, i, j), (16, 16)
            )
    plt.imsave("grid.png", data, cmap="cmo.balance")
    grad_x = img.sobel(data, axis=0)
    grad_y = img.sobel(data, axis=1)
    abs = np.sqrt(grad_x**2 + grad_y**2)
    if save:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["figure.dpi"] = 300
        n = len(abs.flatten())
        plt.xlabel("absolute value of sobel-based gradient")
        plt.ylabel("probability density")
        plt.title(f"Quantum-GRF, $n={n}$")
        plt.imsave("gradient_qgrf.png", abs, cmap="cmo.balance")
        nbins = int(1 + np.ceil(np.log2(n)))
        plt.hist(abs.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("gradhist_qgrf.png")
        plt.clf()
        plt.xlabel("noise values")
        plt.ylabel("probability density")
        plt.title(f"Quantum-GRF, $n={n}$")
        plt.hist(data.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("valhist_qgrf.png")
        plt.clf()
        plt.xlabel("angle of the gradient")
        plt.ylabel("probability density")
        plt.title(f"Quantum-GRF, $n={n}$")
        plt.hist(
            np.arctan2(grad_y, grad_x).flatten(),
            bins=nbins,
            density=True,
            alpha=0.5,
            edgecolor="black",
        )
        ticks = np.linspace(-1, 1, num=5)
        plt.xticks(np.pi * ticks, [rf"${i}\pi$" for i in ticks])
        plt.savefig("graddirhist_qgrf.png")
        plt.clf()
    else:
        return (data, [grad_x, grad_y])


def perlin_stats(file, save=False):
    """
    calculates sobel-filter based gradient for a perlin noise file
    if `save` is set, it will save histograms of the value and gradient distributions
    otherwise, the arrays of values and full gradient will be returned
    """
    data = parse_perlin_csv(file)
    grad_x = img.sobel(data, axis=0)
    grad_y = img.sobel(data, axis=1)
    abs = np.sqrt(grad_x**2 + grad_y**2)
    if save:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["figure.dpi"] = 300
        n = len(abs.flatten())
        plt.xlabel("absolute value of sobel-based gradient")
        plt.ylabel("probability density")
        plt.title(f"Perlin Noise, $n={n}$, 2 octaves per 16 px.")
        plt.imsave("gradientperlin.png", abs)
        nbins = int(1 + np.ceil(np.log2(n)))
        plt.hist(abs.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("gradhistperlin.png")
        plt.clf()
        plt.xlabel("noise values")
        plt.ylabel("probability density")
        plt.title(f"Perlin Noise, $n={n}$, 2 octaves per 16 px.")
        plt.hist(data.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("valhistperlin.png")
        plt.clf()
        plt.xlabel("angle of the gradient")
        plt.ylabel("probability density")
        plt.title(f"Perlin Noise, $n={n}$, 2 octaves per 16 px.")
        plt.hist(
            np.arctan2(grad_y, grad_x).flatten(),
            bins=nbins,
            density=True,
            alpha=0.5,
            edgecolor="black",
        )
        ticks = np.linspace(-1, 1, num=5)
        plt.xticks(np.pi * ticks, [rf"${i}\pi$" for i in ticks])
        plt.savefig("graddirhistperlin.png")
        plt.clf()
    else:
        return (data, [grad_x, grad_y])


def cgrf_stats(folder, save=False, downsample=False):
    """
    calculates sobel-filter based gradient for a collection of classical GRF samples
    if `save` is set, it will save histograms of the value and gradient distributions
    otherwise, the arrays of values and full gradient will be returned
    """
    abs = np.zeros(0)
    angles = np.zeros(0)
    vals = np.zeros(0)
    for i, file in enumerate(os.listdir(folder)):
        data = np.loadtxt(folder + file, delimiter=",")
        if downsample:
            data = data[::15, ::15]
        if i % 5 == 0 and save:
            plt.imsave(f"cgrf_{i}.png", data, cmap="cmo.balance")
        grad_x = img.sobel(data, axis=0)
        grad_y = img.sobel(data, axis=1)
        vals = np.concat((vals, data.flatten()))
        abs = np.concat((abs, np.sqrt(grad_x**2 + grad_y**2).flatten()))
        angles = np.concat((angles, np.arctan2(grad_y, grad_x).flatten()))

    if save:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["figure.dpi"] = 300
        n = len(abs)
        plt.xlabel("absolute value of sobel-based gradient")
        plt.ylabel("probability density")
        plt.title(f"Classical GRF, $n={n}$")
        nbins = int(1 + np.ceil(np.log2(n)))
        plt.hist(abs.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("gradhist_cgrf.png")
        plt.clf()
        plt.xlabel("noise values")
        plt.ylabel("probability density")
        plt.title(f"Classical GRF, $n={n}$")
        plt.hist(data.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("valhist_cgrf.png")
        plt.clf()
        plt.xlabel("angle of the gradient")
        plt.ylabel("probability density")
        plt.title(f"Classical GRF, $n={n}$")
        plt.hist(
            np.arctan2(grad_y, grad_x).flatten(),
            bins=nbins,
            density=True,
            alpha=0.5,
            edgecolor="black",
        )
        ticks = np.linspace(-1, 1, num=5)
        plt.xticks(np.pi * ticks, [rf"${i}\pi$" for i in ticks])
        plt.savefig("graddirhist_cgrf.png")
        plt.clf()
    else:
        return (vals, abs, angles)


def target_value_stats(width, inv_spacing, n, C, xisr, save=False):
    """
    computes `n` samples of the cosine of the multivariate gaussian with gaussian covariance with parameters `C` and `xi^2`
    on the interval [-`width`, `width`] partitioned by `1/inv_spacing`
    saves graphics if `save==True`, returns data points otherwise
    """
    xs = np.linspace(-width, width, num=inv_spacing)
    mean = np.zeros(inv_spacing)
    cov = [[gaussian_covariance(i, j, C, xisr) for j in xs] for i in xs]
    data = np.cos(np.random.multivariate_normal(mean, cov, n))
    if save:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["figure.dpi"] = 300
        n = len(data.flatten())
        nbins = int(1 + np.ceil(np.log2(n)))
        plt.clf()
        plt.xlabel("values")
        plt.ylabel("probability density")
        plt.title(f"Multivariate gaussian sample, $n={n}$")
        plt.hist(data.flatten(), bins=nbins, density=True, alpha=0.5, edgecolor="black")
        plt.savefig("valhist_mvg.png")
    else:
        return data
