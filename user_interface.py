import qiskit as qk
import numpy as np
import pandas as pd
import csv
from qiskit_aer import AerSimulator

from random_generator import pcg6to4, pcg_k_to_l
from util import save_plot
from quantum_circuit import gaussian_random_field, chunked_gaussian_random_field


def parse_chunk(fname, chunk_x, chunk_y):
    d = pd.read_csv(fname)
    str_data = d[str(chunk_x)][chunk_y]
    return np.fromstring(
        str_data.replace("\n", "").replace("[", "").replace("]", ""), sep=" "
    )


def parse_perlin_csv(path):
    data = None
    with open(path, "r") as f:
        r = csv.DictReader(f)
        numlines = sum(1 for _ in r)
        data = np.zeros(numlines)
        f.seek(0)
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            data[i] = row["out"]
        data = np.reshape(data, (int(np.sqrt(numlines)), int(np.sqrt(numlines))))
    return data


def probabilities_from_result(result):

    raw_counts = result.get_counts()
    # We filter the results where the bit |b> is zero
    counts_b_zero = {
        int(k.split(" ")[1], 2): v
        for k, v in raw_counts.items()
        if k.split(" ")[0] == "0"
    }
    # Similarly for one
    counts_b_one = {
        int(k.split(" ")[1], 2): v
        for k, v in raw_counts.items()
        if k.split(" ")[0] == "1"
    }

    data_b_zero = np.zeros(2**8)
    data_b_one = np.zeros(2**8)
    # data_b_one = np.zeros(2**8)
    for i in range(2**8):
        if i in counts_b_zero:
            data_b_zero[i] = counts_b_zero[i]
        if i in counts_b_one:
            data_b_one[i] = counts_b_one[i]

    # The (approx) probability of finding |j> with |b> = |0> or |1> respectively is then:
    for i in range(2**8):
        total_counts_i = data_b_one[i] + data_b_zero[i]
        if total_counts_i != 0:
            data_b_zero[i] /= float(total_counts_i)
            data_b_one[i] /= float(total_counts_i)

    if sum(data_b_zero + data_b_one) != 2**8:
        print(
            "normalization warning, total is "
            + str(sum(data_b_zero + data_b_one))
            + " instead of 256"
        )
        print(
            "not every pixel had counts, most likely, the number of shots was too little"
        )
    return (data_b_zero, data_b_one)


def run_default_circuit(backend, n_shots: int, post_process=False, pcg_params=[17, 3]):
    """
    Runs the default Deiml et al. circuit, except for the initial R_y and halving the rotation angle
    to avoid angles getting too big.
    Simulated with `backend` and measurements obtained with `n_shots` shots
    If `post_process` is set to `True`, the probability of measuring `|b> = 0` will be converted have values in [-1, 1]
    By default, the raw probability will be returned.

    """
    pcg = pcg6to4(pcg_params[0], pcg_params[1])
    qc_grf = gaussian_random_field(pcg)

    j = qk.QuantumRegister(8, name="j")
    b = qk.QuantumRegister(1, name="b")
    a_boundary = qk.QuantumRegister(2, name="ab")
    a_overflow = qk.QuantumRegister(1, name="ao")
    a_sq = qk.QuantumRegister(6, name="asq")
    a_iter = qk.QuantumRegister(2, name="ai")
    class_j = qk.ClassicalRegister(8, name="j_out")
    class_b = qk.ClassicalRegister(1, name="b_out")

    qc = qk.QuantumCircuit(
        j,
        b,
        a_boundary,
        a_overflow,
        a_sq,
        a_iter,
        class_j,
        class_b,
    )
    qc.h(j)
    qc.append(
        qc_grf,
        j[:] + b[:] + a_boundary[:] + a_overflow[:] + a_sq[:] + a_iter[:],
    )
    qc.measure(j, class_j)
    qc.measure(b, class_b)

    circ = qk.transpile(qc, backend)
    result = backend.run(circ, shots=n_shots).result()
    p_zero, p_one = probabilities_from_result(result)
    if post_process:
        return (p_zero - 1.0 / 2) * 2
    return p_zero


def run_chunk_circuit(
    chunk_x: int,
    chunk_y: int,
    max_chunks: int,
    backend,
    n_shots: int,
    post_process=False,
    pcg_params=[17, 3],
):
    """
    Runs the circuit for a 16x16 "chunk", given by the coordinates (`chunk_y`, `chunk_x`).
    The maximum value for either coordinate is limited by `max_chunks`, which determines how many extra bits will be used
    for the chunk address space.
    Simulated by the given `backend`.with `n_shots` shots.
    If `post_process` is set to `True`, the probability of measuring `|b> = 0` will be converted have values in [-1, 1]
    By default, the raw probability will be returned.
    """
    assert chunk_x < max_chunks
    assert chunk_y < max_chunks
    chunk_bits = int(np.ceil(np.log2(max_chunks)))

    pcg = pcg_k_to_l(2 * chunk_bits + 6, 4, pcg_params[0], pcg_params[1])

    qc_grf = chunked_gaussian_random_field(chunk_x, chunk_y, chunk_bits, pcg)

    j = qk.QuantumRegister(8, name="j")
    k = qk.QuantumRegister(2 * chunk_bits, name="k")
    b = qk.QuantumRegister(1, name="b")
    a_boundary = qk.QuantumRegister(2, name="ab")
    a_overflow = qk.QuantumRegister(1, name="ao")
    square1 = qk.QuantumRegister(2 * chunk_bits + 6, name="sq1")
    a_iter = qk.QuantumRegister(2, name="ai")
    class_j = qk.ClassicalRegister(8, name="j_out")
    class_b = qk.ClassicalRegister(1, name="b_out")

    qc = qk.QuantumCircuit(
        j,
        k,
        b,
        a_boundary,
        a_overflow,
        square1,
        a_iter,
        class_j,
        class_b,
    )
    qc.h(j)
    qc.append(
        qc_grf,
        j[:] + k[:] + b[:] + a_boundary[:] + a_overflow[:] + square1[:] + a_iter[:],
    )
    qc.measure(j, class_j)
    qc.measure(b, class_b)

    circ = qk.transpile(qc, backend)
    result = backend.run(circ, shots=n_shots).result()
    p_zero, p_one = probabilities_from_result(result)
    if post_process:
        return (p_zero - 1.0 / 2) * 2
    return p_zero
