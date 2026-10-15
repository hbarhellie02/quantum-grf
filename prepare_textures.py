import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from user_interface import parse_chunk


def get_full_data(file, save=False):
    chunks_x = 8
    chunks_y = 8
    data = np.zeros((16 * chunks_x, 16 * chunks_y))
    for i in range(chunks_x):
        for j in range(chunks_y):
            data[16 * i : 16 * (i + 1), 16 * j : 16 * (j + 1)] = np.reshape(
                parse_chunk(file, i, j), (16, 16)
            )
    if save:
        plt.gray()
        plt.imsave("heightmap.png", data)
        plt.imsave("colors.png", data, cmap="gist_earth")
    else:
        return data


def get_interpolated_data_from_file(file, n_chunks, save=False):
    data = np.zeros((16 * n_chunks, 16 * n_chunks))
    for i in range(n_chunks):
        for j in range(n_chunks):
            data[16 * i : 16 * (i + 1), 16 * j : 16 * (j + 1)] = np.reshape(
                parse_chunk(file, i, j), (16, 16)
            )
    xs = np.linspace(0, 128, 16 * n_chunks)
    interp = RegularGridInterpolator((xs, xs), data)
    outx = np.array(range(128))
    xs, ys = np.meshgrid(outx, outx)
    outd = interp((xs, ys))
    if save:
        plt.gray()
        plt.imsave("heightmap.png", outd)
        plt.imsave("colors.png", outd, cmap="gist_earth")
    else:
        return outd


def get_interpolated_data_from_world(world, n_chunks, save=False):
    xs = np.linspace(0, 128, 16 * n_chunks)
    interp = RegularGridInterpolator((xs, xs), world[: 16 * n_chunks, : 16 * n_chunks])
    outx = np.array(range(128))
    xs, ys = np.meshgrid(outx, outx)
    outd = interp((xs, ys))
    if save:
        plt.gray()
        plt.imsave("heightmap.png", outd)
        plt.imsave("colors.png", outd, cmap="gist_earth")
    else:
        return outd


def prepare_assets_from_file(file, n_octaves, base_amplitude):
    assert n_octaves <= 4
    data = np.zeros((128, 128))
    for i in range(n_octaves):
        if i == 3:
            data += get_full_data(file)
        else:
            data += base_amplitude ** (3 - i) * get_interpolated_data_from_file(
                file, 2**i
            )
    plt.gray()
    plt.imsave("heightmap.png", data)
    plt.imsave("colors.png", data, cmap="gist_earth")


def prepare_assets_from_world(world, n_octaves, base_amplitude, save_map=False):
    assert n_octaves <= 4
    data = np.zeros((128, 128))
    for i in range(n_octaves):
        if i == 3:
            data += world
        else:
            data += base_amplitude ** (
                n_octaves - 1 - i
            ) * get_interpolated_data_from_world(world, 2**i)
    plt.gray()
    plt.imsave("heightmap.png", data)
    plt.imsave("colors.png", data, cmap="gist_earth")
    if save_map:
        plt.imsave("world.png", world, cmap="cmo.balance")
