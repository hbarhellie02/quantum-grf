import qiskit as qk
import numpy as np
from qiskit_aer import AerSimulator

from random_generator import pcg6to4, pcg_k_to_l
from util import save_plot
from quantum_circuit import gaussian_random_field, chunked_gaussian_random_field
from reproducibility_test import probabilities_from_result


def run_default_circuit(backend, n_shots: int, post_process=False):
    """
    Runs the default Deiml et al. circuit, except for the initial R_y and halving the rotation angle
    to avoid angles getting too big.
    Simulated with `backend` and measurements obtained with `n_shots` shots
    If `post_process` is set to `True`, the probability of measuring `|b> = 0` will be converted have values in [-1, 1]
    By default, the raw probability will be returned.

    """
    pcg = pcg6to4(17, 3)
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
    chunk_bits = int(2 ** (np.ceil(np.log2(max_chunks))))

    pcg = pcg_k_to_l(2 * chunk_bits + 6, 4, 17, 3)

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
