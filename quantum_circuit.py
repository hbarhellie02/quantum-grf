# Copyright (C) 2025 Matthias Deiml, Daniel Peterseim - All rights reserved

import qiskit as qk
import numpy as np
from qiskit_aer import AerSimulator

from random_generator import pcg6to4, constant_add, pcg_k_to_l
from util import save_plot

backend = AerSimulator(method="statevector")

print("Finding polynomial approximation for exp(-x**2)...")
xs = np.linspace(-8, 8, 200)
ys = np.exp(-((xs / 4) ** 2))
p = np.polyfit(xs, ys, 4)
p = np.polynomial.Polynomial(p[::-2] * 0.5)


# Circuit for computing the square of the lower three bits (as an unsigned integer) into the combined six.
# The upper bits must contain zero
# Example: the state prepared as '000011'=3 will yield '001001'=9
#
# For Qiskit reasons, in a register composed of two s = a[:] + b[:], the lower bits are in a, i.e b[:] must be '000'
def qc_sq() -> qk.QuantumCircuit():
    qc = qk.QuantumCircuit(6, name="sq")
    qc.ccx(1, 2, 5)
    qc.ccx(0, 5, 2, ctrl_state="10")
    qc.cx(2, 4)
    qc.cx(1, 2)
    qc.ccx(0, 2, 3)
    qc.cx(2, 1)
    qc.cx(3, 5)
    qc.ccx(5, 4, 2)
    qc.cx(3, 5)
    qc.cx(3, 2)
    qc.cx(4, 2)
    qc.cx(4, 1)
    return qc


def n_constant_add_or_sub(n, c):
    """
    Circuit that adds a (possibly negative) constant to a 3 qubit register
    """
    qc = qk.QuantumCircuit(n)

    qc.append(qk.circuit.library.QFT(n, do_swaps=False), range(n))
    if c > 0:
        qc.append(constant_add(n, abs(c), control_bits=0), range(n))
    else:
        qc.append(constant_add(n, abs(c), control_bits=0).inverse(), range(n))
    qc.append(qk.circuit.library.QFT(n, do_swaps=False).inverse(), range(n))

    return qc


def constant_add_or_sub(c):
    return n_constant_add_or_sub(3, c)


def rotate_by_kernel():
    """
    Circuit that rotates the register b around the y axis by an angle that is
    the kernel applied to the vector stored in dx and dy.
    """
    dx = qk.QuantumRegister(3, name="dx")
    dy = qk.QuantumRegister(3, name="dy")
    b = qk.QuantumRegister(1, name="b")
    a_overflow = qk.QuantumRegister(1, name="ao")
    a_sq = qk.QuantumRegister(6, name="as")
    qc = qk.QuantumCircuit(dx, dy, b, a_overflow, a_sq, name="rot")

    sqx = dx[:] + a_sq[0:3]
    qc.append(qc_sq(), sqx)
    sqy = dy[:] + a_sq[3:6]
    qc.append(qc_sq(), sqy)
    qc.append(
        qk.circuit.library.DraperQFTAdder(6, kind="half"),
        sqy + sqx + a_overflow[:],  # sqx now contains sqy+sqx
    )
    qc.append(qc_sq().inverse(), sqy)
    qc.x(a_overflow)

    qc.cry(2 * p.coef[0], a_overflow, b)

    for i in range(6):
        qc.mcry(2 * p.coef[1] * 2**i, [a_overflow[0], sqx[i]], b)

    sq = sqx[2:5] + a_sq[3:6]
    qc.append(qc_sq(), sq)

    for i in range(6):
        qc.mcry(2 * p.coef[2] * 2 ** (i + 4), [a_overflow[0], sq[i]], b)

    qc.append(qc_sq().inverse(), sq)
    qc.mcry(2 * p.coef[2] * 2 ** (10), [a_overflow[0], sqx[5]], b)

    for i in range(3):
        qc.mcry(
            2 * p.coef[2] * 2 ** (i + 7),
            [a_overflow[0], sqx[5], sqx[2 + i]],
            b,
        )

    qc.x(a_overflow)
    qc.append(qc_sq(), sqy)
    qc.append(
        qk.circuit.library.DraperQFTAdder(6, kind="half").inverse(),
        sqy + sqx + a_overflow[:],
    )
    qc.append(qc_sq().inverse(), sqy)
    qc.append(qc_sq().inverse(), sqx)

    return qc


def n_rotate_by_kernel(dx_size):
    """
    Circuit that rotates the register b around the y axis by an angle that is
    the kernel applied to the vector stored in dx and dy.
    """
    assert False, "Not ready to use!"
    dx = qk.QuantumRegister(dx_size, name="dx")
    dy = qk.QuantumRegister(dx_size, name="dy")
    b = qk.QuantumRegister(1, name="b")
    a_overflow = qk.QuantumRegister(1, name="ao")
    a_sq = qk.QuantumRegister(2 * dx_size, name="as")
    qc = qk.QuantumCircuit(dx, dy, b, a_overflow, a_sq, name="rot")

    sqx = dx[:] + a_sq[0:dx_size]
    qc.append(qc_sq(), sqx)
    sqy = dy[:] + a_sq[dx_size:]
    qc.append(qc_sq(), sqy)
    qc.append(
        qk.circuit.library.DraperQFTAdder(2 * dx_size, kind="half"),
        sqy + sqx + a_overflow[:],
    )
    qc.append(qc_sq().inverse(), sqy)
    qc.x(a_overflow)

    qc.cry(2 * p.coef[0], a_overflow, b)

    for i in range(2 * dx_size):
        qc.mcry(2 * p.coef[1] * 2**i, [a_overflow[0], sqx[i]], b)

    sq = sqx[2:5] + a_sq[3:6]
    qc.append(qc_sq(), sq)

    for i in range(2 * dx_size):
        qc.mcry(2 * p.coef[2] * 2 ** (i + 4), [a_overflow[0], sq[i]], b)

    qc.append(qc_sq().inverse(), sq)
    qc.mcry(2 * p.coef[2] * 2 ** (10), [a_overflow[0], sqx[5]], b)

    for i in range(3):
        qc.mcry(
            2 * p.coef[2] * 2 ** (i + 7),
            [a_overflow[0], sqx[5], sqx[2 + i]],
            b,
        )

    qc.x(a_overflow)
    qc.append(qc_sq(), sqy)
    qc.append(
        qk.circuit.library.DraperQFTAdder(6, kind="half").inverse(),
        sqy + sqx + a_overflow[:],
    )
    qc.append(qc_sq().inverse(), sqy)
    qc.append(qc_sq().inverse(), sqx)

    return qc


def gaussian_random_field(pcg):
    j = qk.QuantumRegister(8, name="j")
    b = qk.QuantumRegister(1, name="b")
    a_boundary = qk.QuantumRegister(2, name="ab")
    a_overflow = qk.QuantumRegister(1, name="ao")
    a_sq = qk.QuantumRegister(6, name="as")
    a_iter = qk.QuantumRegister(2, name="ai")

    qc = qk.QuantumCircuit(j, b, a_boundary, a_overflow, a_sq, a_iter, name="grf")

    # Loop over noise variables in the neighbourhood of
    # the coordinates stored in j
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # For dx = 0 and dy = 0 we consider the noise variable with the
            # largest coordinates below the coordinates in j. This index of
            # this variable is given by the most significant bits of each
            # coordinate. For other dx and dy the node with the offset (dx, dy)
            # is considered.

            # Here, we first compute the index of this node into the upper bits
            # of j.
            qc.append(constant_add_or_sub(dx), j[2:4] + [a_boundary[0]])
            qc.append(constant_add_or_sub(dy), j[6:8] + [a_boundary[1]])

            # We then iterate over the random bits used in the definition of the
            # noise variable
            for k in range(4):
                # The corresponding random bit is computed, and b is flipped if
                # the bit is 1.
                qc.append(
                    pcg, j[2:4] + j[6:8] + a_boundary[:] + a_sq[:] + [a_overflow[0]]
                )
                qc.cx(a_sq[k], b)
                qc.append(
                    pcg.inverse(),
                    j[2:4] + j[6:8] + a_boundary[:] + a_sq[:] + [a_overflow[0]],
                )

                # We compute the difference in x direction between the
                # coordinates of the noise variable and the original j. This is
                # a 3 bit integer stored in j[0], j[1] and a_iter[0].
                if dx == -1:
                    qc.x(a_iter[0])
                elif dx == 1:
                    qc.ccx(j[0], j[1], a_iter[0], ctrl_state=0)
                    qc.cx(j[0], j[1])

                # We do the same for the y direction, storing the result in
                # j[4], j[5] and a_iter[1].
                if dy == -1:
                    qc.x(a_iter[1])
                elif dy == 1:
                    qc.ccx(j[4], j[5], a_iter[1], ctrl_state=0)
                    qc.cx(j[4], j[5])

                # We then rotate b by the kernel evaluted at the computed
                # difference.
                qc.append(
                    rotate_by_kernel(),
                    j[0:2]
                    + [a_iter[0]]
                    + j[4:6]
                    + [a_iter[1]]
                    + b[:]
                    + a_overflow[:]
                    + a_sq[:],
                )

                # And uncompute everything
                if dy == -1:
                    qc.x(a_iter[1])
                elif dy == 1:
                    qc.cx(j[4], j[5])
                    qc.ccx(j[4], j[5], a_iter[1], ctrl_state=0)
                if dx == -1:
                    qc.x(a_iter[0])
                elif dx == 1:
                    qc.cx(j[0], j[1])
                    qc.ccx(j[0], j[1], a_iter[0], ctrl_state=0)

                # And potentially flip b back
                qc.append(
                    pcg, j[2:4] + j[6:8] + a_boundary[:] + a_sq[:] + [a_overflow[0]]
                )
                qc.cx(a_sq[k], b)
                qc.append(
                    pcg.inverse(),
                    j[2:4] + j[6:8] + a_boundary[:] + a_sq[:] + [a_overflow[0]],
                )
                # The correctness of this can be checked in the following
                # way: If the random bit is 0, then the angle of the rotation
                # introduced by rotate_by_kernel is just the value kernel.
                # If the random bit is 1, b if flipped before and after
                # rotate_by_kernel, causing a rotation in the opposite
                # direction.

            # Considerin all iteration of the loop, b has now been rotated by an
            # angle that is the summand in the quadrature corresponding to the
            # noise variable with the offset (dx, dy).

            qc.append(constant_add_or_sub(dy).inverse(), j[6:8] + [a_boundary[1]])
            qc.append(constant_add_or_sub(dx).inverse(), j[2:4] + [a_boundary[0]])

    # Repeating this procedure leads to a rotation by an angle that is the
    # entire quadrature. The value of the transformed random field is now
    # just the coefficient of the state where b is 1. All other registers were
    # returned to their initial state.

    # Finally, we flip b once more so that the result correspond to the 0 state.
    # (This is standard for block encodings.)

    qc.x(b)
    return qc


def n_gaussian_random_field(nbits: int, pcg) -> qk.QuantumCircuit:
    # Must be an even number to be able to produce a nice square grid.
    assert nbits % 2 == 0

    half_bits = nbits // 2

    pcg = pcg_k_to_l()

    j = qk.QuantumRegister(nbits, name="j")
    b = qk.QuantumRegister(1, name="b")
    a_boundary = qk.QuantumRegister(2, name="ab")
    a_overflow = qk.QuantumRegister(1, name="ao")
    a_sq = qk.QuantumRegister(6, name="as")
    a_iter = qk.QuantumRegister(2, name="ai")

    qc = qk.QuantumCircuit(
        j, b, a_boundary, a_overflow, a_sq, a_iter, name=f"{nbits}-bit-grf"
    )

    # Loop over noise variables in the neighbourhood of
    # the coordinates stored in j
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # For dx = 0 and dy = 0 we consider the noise variable with the
            # largest coordinates below the coordinates in j. This index of
            # this variable is given by the most significant bits of each
            # coordinate. For other dx and dy the node with the offset (dx, dy)
            # is considered.

            # Here, we first compute the index of this node into the upper bits of each half
            # of j.
            qc.append(
                constant_add_or_sub(dx), j[half_bits - 2 : half_bits] + [a_boundary[0]]
            )
            qc.append(constant_add_or_sub(dy), j[nbits - 2 : nbits] + [a_boundary[1]])

            # We then iterate over the random bits used in the definition of the
            # noise variable
            for k in range(4):
                # The corresponding random bit is computed, and b is flipped if
                # the bit is 1.
                qc.append(
                    pcg,
                    j[half_bits - 2 : half_bits]
                    + j[nbits - 2 : nbits]
                    + a_boundary[:]
                    + a_sq[:]
                    + [a_overflow[0]],
                )
                qc.cx(a_sq[k], b)
                qc.append(
                    pcg.inverse(),
                    j[half_bits - 2 : half_bits]
                    + j[nbits - 2 : nbits]
                    + a_boundary[:]
                    + a_sq[:]
                    + [a_overflow[0]],
                )

                # We compute the difference in x direction between the
                # coordinates of the noise variable and the original j. This is
                # a 3 bit integer stored in j[0], j[1] and a_iter[0].
                if dx == -1:
                    qc.x(a_iter[0])
                elif dx == 1:
                    qc.ccx(j[0], j[1], a_iter[0], ctrl_state=0)
                    qc.cx(j[0], j[1])

                # We do the same for the y direction, storing the result in
                # j[4], j[5] and a_iter[1].
                if dy == -1:
                    qc.x(a_iter[1])
                elif dy == 1:
                    qc.ccx(j[half_bits], j[half_bits + 1], a_iter[1], ctrl_state=0)
                    qc.cx(j[half_bits], j[half_bits + 1])

                # We then rotate b by the kernel evaluted at the computed
                # difference.
                qc.append(
                    rotate_by_kernel(),
                    j[0:2]
                    + [a_iter[0]]
                    + j[half_bits : half_bits + 2]
                    + [a_iter[1]]
                    + b[:]
                    + a_overflow[:]
                    + a_sq[:],
                )

                # And uncompute everything
                if dy == -1:
                    qc.x(a_iter[1])
                elif dy == 1:
                    qc.cx(j[half_bits], j[half_bits + 1])
                    qc.ccx(j[half_bits], j[half_bits + 1], a_iter[1], ctrl_state=0)
                if dx == -1:
                    qc.x(a_iter[0])
                elif dx == 1:
                    qc.cx(j[0], j[1])
                    qc.ccx(j[0], j[1], a_iter[0], ctrl_state=0)

                # And potentially flip b back
                qc.append(
                    pcg,
                    j[half_bits - 2 : half_bits]
                    + j[nbits - 2 : nbits]
                    + a_boundary[:]
                    + a_sq[:]
                    + [a_overflow[0]],
                )
                qc.cx(a_sq[k], b)
                qc.append(
                    pcg.inverse(),
                    j[half_bits - 2 : half_bits]
                    + j[nbits - 2 : nbits]
                    + a_boundary[:]
                    + a_sq[:]
                    + [a_overflow[0]],
                )
                # The correctness of this can be checked in the following
                # way: If the random bit is 0, then the angle of the rotation
                # introduced by rotate_by_kernel is just the value kernel.
                # If the random bit is 1, b if flipped before and after
                # rotate_by_kernel, causing a rotation in the opposite
                # direction.

            # Considerin all iteration of the loop, b has now been rotated by an
            # angle that is the summand in the quadrature corresponding to the
            # noise variable with the offset (dx, dy).

            qc.append(
                constant_add_or_sub(dy).inverse(),
                j[nbits - 2 : nbits] + [a_boundary[1]],
            )
            qc.append(
                constant_add_or_sub(dx).inverse(),
                j[half_bits - 2 : half_bits] + [a_boundary[0]],
            )

    # Repeating this procedure leads to a rotation by an angle that is the
    # entire quadrature. The value of the transformed random field is now
    # just the coefficient of the state where b is 1. All other registers were
    # returned to their initial state.

    # Finally, we flip b once more so that the result correspond to the 0 state.
    # (This is standard for block encodings.)

    qc.x(b)
    return qc


if __name__ == "__main__":
    pcg = pcg6to4(17, 3)

    n = 10

    qc_grf = n_gaussian_random_field(n, pcg)

    j = qk.QuantumRegister(n, name="j")
    b = qk.QuantumRegister(1, name="b")
    b1 = qk.QuantumRegister(1, name="b1")
    a_boundary = qk.QuantumRegister(2, name="ab")
    a_overflow = qk.QuantumRegister(1, name="ao")
    a_sq = qk.QuantumRegister(
        6, name="as"
    )  # Are these the m additional bits for computing theta?
    a_iter = qk.QuantumRegister(2, name="ai")

    qc = qk.QuantumCircuit(j, b, a_boundary, a_overflow, a_sq, a_iter)
    qc.h(j)
    qc.append(qc_grf, j[:] + b[:] + a_boundary[:] + a_overflow[:] + a_sq[:] + a_iter[:])
    qc.save_statevector()

    print("Building circuit for plot...")
    circ = qk.transpile(qc, backend)

    print("Simulating circuit for plot...")
    result = backend.run(circ).result()
    data = result.get_statevector().data[0 : 2**n].real
    data = np.reshape(data, (2 ** (n // 2), 2 ** (n // 2)), copy=False)

    save_plot(data, "quantum_circuit_cos.png")

    samples = 32
    cov = 0

    for s in range(samples):
        pcg = pcg6to4(17, 1 + s * 2)

        qc_grf = gaussian_random_field(pcg)

        # This circuit corresponds to measuring the quantity of interest
        # E[Z_left Z_right] as defined in Section 6.3.

        qc = qk.QuantumCircuit(j, b, b1, a_boundary, a_overflow, a_sq, a_iter)

        qc.h(j[0:3] + j[4:8])
        qc.append(
            qc_grf, j[:] + b[:] + a_boundary[:] + a_overflow[:] + a_sq[:] + a_iter[:]
        )
        qc.h(j[0:3] + j[4:8])
        qc.mcx(j[:] + b[:], b1, ctrl_state=0)
        qc.x(b1)
        qc.h(j[0:3] + j[4:8])
        qc.x(j[3])
        qc.append(
            qc_grf, j[:] + b[:] + a_boundary[:] + a_overflow[:] + a_sq[:] + a_iter[:]
        )
        qc.x(j[3])
        qc.h(j[0:3] + j[4:8])

        qc.save_statevector()

        print(f"Building circuit for sample {s + 1}/{samples}...")
        circ = qk.transpile(qc, backend)
        print(f"Simulating circuit for sample {s + 1}/{samples}...")
        result = backend.run(circ).result()

        cov_sample = np.abs(result.get_statevector().data[0])
        print(f"Resulting covariance: {cov_sample}")
        cov += cov_sample

    print(f"Average coavariance: {cov / samples}")
