# Copyright (C) 2025 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
import numpy as np
import qiskit_aer
from qiskit.circuit.library import QFT

from util import save_plot


def CInc(N):
    """
    Circuit implementing a increment operation of the second N-qubit register
    controlled by the first quibt register.

    Optimized variants are implemented for N = 1, 2
    """
    x = qk.QuantumRegister(1, name="x")
    y = qk.QuantumRegister(N, name="y")

    qc = qk.QuantumCircuit(x, y, name="c_inc")

    if N == 1:
        qc.cx(x, y)
    elif N == 2:
        qc.ccx(x, y[0], y[1])
        qc.cx(x, y[0])
    else:
        qc.append(QFT(N, do_swaps=False).to_gate(), y[:])
        for j in range(N):
            lam = np.pi / (2**j)
            qc.cp(lam, x, y[j])
        qc.append(QFT(N, do_swaps=False).inverse().to_gate(), y[:])

    return qc


def CRShift(LOG_N, DIM, uncompute_control=False):
    """
    Circuit implementing a bitwise right shift of the first register by the
    second register.

    The first register is interpreted as consisting of DIM registers with N
    qubits each, all of which are shifted.
    """

    N = 2**LOG_N
    x = qk.QuantumRegister(N * DIM, name="x")
    y = qk.QuantumRegister(LOG_N, name="y")

    qc = qk.QuantumCircuit(x, y, name="shift")
    if LOG_N == 1:
        for d in range(DIM):
            qc.cswap(y[0], x[2 * d], x[2 * d + 1])
    else:
        for d in range(DIM):
            for i in range(0, N, 2):
                qc.cswap(y[0], x[N * d + i], x[N * d + i + 1])
        next = CRShift(LOG_N - 1, DIM)
        qc.append(next, x[::2] + y[1:])
        qc.append(CInc(LOG_N - 1), y[:])

        qc.append(next, x[1::2] + y[1:])
        if uncompute_control:
            qc.append(CInc(LOG_N - 1).inverse(), y[:])

    return qc.decompose(["c_inc", "shift"])


def mod_inv(n, m):
    """
    Modular inverse of n with respect to the modulus 2**m
    """
    return pow(n, -1, mod=2**m)


def xorshift(bits: int, shift: int) -> qk.QuantumCircuit:
    """
    Circuit that xors the highest (bits - shift) bits onto the lowest
    (bits - shift) bits
    """
    assert shift < bits

    qc = qk.QuantumCircuit(bits, name=f"xs({shift})")

    for i in range(bits - shift):
        qc.cx(i + shift, i)

    return qc


def lcg(bits: int, bits_step: int, m: int, c: int) -> qk.QuantumCircuit:
    """
    Circuit that implement skipping for a linear congruential generator (LCG)

    m and c correspond to the parameters of the lcg, then calculations are
    preformed modulo 2**bits, and bits_step determines the number of bits for
    the size of the skip.
    """
    n = qk.QuantumRegister(bits_step, name="n")
    x = qk.QuantumRegister(bits, name="x")
    a = qk.QuantumRegister(1, name="a")

    qc = qk.QuantumCircuit(n, x, a, name="lcg")

    acc = m
    acc_c = c
    for i in range(bits_step):
        if i % 2 == 0:
            qc.append(constant_mult(bits, acc), [n[i]] + x[:] + a[:])
            qc.append(constant_add(bits, acc_c), [n[i]] + x[:])
        else:
            factor = mod_inv(acc, bits)
            shift = (acc_c * factor) % (2**bits)
            qc.append(constant_add(bits, shift), [n[i]] + x[:])
            qc.append(constant_mult(bits, factor).inverse(), [n[i]] + x[:] + a[:])
        acc_c = (acc_c + acc * acc_c) % (2**bits)
        acc = (acc**2) % (2**bits)

    if bits_step % 2 == 1:
        qc.append(QFT(bits, do_swaps=False).inverse(), x[:])

    return qc


def constant_add(bits: int, m: int, control_bits=1) -> qk.QuantumCircuit:
    """
    Circuit for adding a constant to a register of `bits` qubits

    If control_bits is not zero, this returns a controlled gate.
    """
    if control_bits == 0:
        x = qk.QuantumRegister(bits, name="x")
        qc = qk.QuantumCircuit(x, name="+" + str(m))
    elif control_bits == 1:
        c = qk.QuantumRegister(1, name="c")
        x = qk.QuantumRegister(bits, name="x")
        qc = qk.QuantumCircuit(c, x, name="+" + str(m))
    else:
        c = qk.QuantumRegister(control_bits, name="c")
        x = qk.QuantumRegister(bits, name="x")
        a = qk.QuantumRegister(1, name="a")
        qc = qk.QuantumCircuit(c, x, a, name="+" + str(m))

    phases = np.zeros(bits)
    nonzero = 0

    for k in range(bits):
        sum = 0
        for j in range(k + 1):
            if (m >> j) & 1 == 1:
                phases[k] += 2**j
        if sum != 0:
            nonzero += 1

    ctrl = None
    if control_bits == 1:
        ctrl = c
    elif control_bits > 0 and nonzero >= 6:
        qc.mcx(c, a)
        ctrl = a
    for k in range(bits):
        if phases[k] == 0:
            continue
        if ctrl is None:
            qc.p(phases[k] * np.pi / 2**k, x[k])
        else:
            qc.mcp(phases[k] * np.pi / 2**k, ctrl, x[k])
    if control_bits > 1 and nonzero >= 6:
        qc.mcx(c, a)

    return qc


def constant_mult(bits: int, m: int) -> qk.QuantumCircuit:
    """
    Circuit for multiplying an odd constant to a register of `bits` qubits
    """
    assert m % 2 == 1

    # Assumes that the high `bits - 1` bits of x encode the Fourier transform of
    # the actual bits.

    c = qk.QuantumRegister(1, name="c")
    x = qk.QuantumRegister(bits, name="x")
    a = qk.QuantumRegister(1, name="a")

    qc = qk.QuantumCircuit(c, x, a, name=f"*{m}")

    qc.h(x[-1])
    for i in reversed(range(bits - 1)):
        qc.append(
            constant_add(bits - i - 1, m >> 1, control_bits=2),
            c[:] + [x[i]] + x[i + 1 :] + a[:],
        )

        for k in range(1, bits - i):
            qc.cp(2 * np.pi / 2 ** (k + 1), x[i + k], x[i])
        qc.h(x[i])

    return qc


def pcg6to4(m, c) -> qk.QuantumCircuit:
    """
    Implementation of a permuted congruential generator (PCG) with state size 6
    and which outputs a 4 bit random integer.
    """
    n = qk.QuantumRegister(6, name="n")
    x = qk.QuantumRegister(6, name="x")
    a = qk.QuantumRegister(1, name="a")

    circ = qk.QuantumCircuit(n, x, a, name="pcg")
    circ.append(lcg(6, 6, m, c), n[:] + x[:] + a[:])
    circ.append(xorshift(6, 2), x[:])
    circ.append(CRShift(2, 1), x[:4] + x[-2:])

    return circ


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random

    b1 = 6
    b2 = 6

    m = 17
    c = 1 + random.randint(0, 2 ** (b1 - 1) - 1) * 2
    print(c)

    output_bits = 4

    assert output_bits < b1

    backend = qiskit_aer.AerSimulator()

    def sample(seed: int):
        n = qk.QuantumRegister(b2, name="n")
        x = qk.QuantumRegister(b1, name="x")
        a = qk.QuantumRegister(1, name="a")
        out = qk.ClassicalRegister(output_bits, name="out")

        qcs = []

        qc_pcg = pcg6to4(m, c)

        for i in range(2**b2):
            circ = qk.QuantumCircuit(n, x, a, out)
            circ.prepare_state(i, n)
            circ.prepare_state(seed, x)

            circ.append(qc_pcg, n[:] + x[:] + a[:])

            circ.measure(x[:output_bits], out)

            qcs.append(circ)

        qcs = qk.transpile(qcs, backend, optimization_level=0)

        # Only need one shot since the cirucit is deterministic
        result = backend.run(qcs, shots=1).result()

        data = np.zeros(output_bits * 2**b2, dtype=np.uint8)

        for i, qc in enumerate(qcs):
            data[i * output_bits : (i + 1) * output_bits] = np.array(
                [int(c) for c in reversed(list(result.get_counts(qc).keys())[0])]
            )

        return data

    seed = random.randint(0, 2**b1)

    data = sample(seed)

    data.shape = (16, 16)

    # data = np.random.binomial(1, 0.5, size=(32, 32))

    # from util import save_image
    save_plot(-1.0 * data, "pcg.png", max=1.5)
