import qiskit as qk
import numpy as np
from qiskit_aer import AerSimulator
import pandas as pd
import csv

from util import save_plot
from user_interface import run_chunk_circuit


if __name__ == "__main__":
    backend = AerSimulator(
        method="matrix_product_state", mps_omp_threads=14
    )  # method="statevector")  # method=sampler ? or default
    pcg_params = [21, 3]
    x_chunks = 2
    y_chunks = 8
    max_chunks = 8
    shots = 2**14  # yields an average of 64 shots per pixel
    df = pd.DataFrame(np.zeros((x_chunks, y_chunks)), dtype=object)
    df.attrs = {
        "PCG parameters": f"{pcg_params[0]}, {pcg_params[1]}",
        "max. chunks": max_chunks,
        "shots": shots,
    }
    filename = f"values_{x_chunks}_{y_chunks}_chunks.csv"

    for chunk_x in range(x_chunks):
        for chunk_y in range(y_chunks):
            data = run_chunk_circuit(
                chunk_x,
                chunk_y,
                max_chunks,
                backend,
                shots,
                post_process=True,
                pcg_params=pcg_params,
            )
            data = np.reshape(data, (16, 16), copy=False)
            df.at[chunk_x, chunk_y] = data
            save_plot(data, f"shifted-modified-chunk-{chunk_y}_{chunk_x}.png")
            save_plot(-data, f"shifted-modified-chunk-{chunk_y}_{chunk_x}_neg.png")
    df.to_csv(filename)
    with open(filename, mode="a") as f:
        w = csv.DictWriter(f, fieldnames=list(df.attrs.keys()))
        w.writeheader()
        w.writerow(df.attrs)
