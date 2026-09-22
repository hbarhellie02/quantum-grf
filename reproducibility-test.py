import qiskit as qk
import numpy as np
from qiskit_aer import AerSimulator

from random_generator import pcg6to4, pcg_k_to_l
from util import save_plot
from quantum_circuit import gaussian_random_field, chunked_gaussian_random_field


def probabilities_from_result(result):

    raw_counts = result.get_counts()
    # We filter the results where the bit |b> is zero. In the algorithm in the paper this correspronds to measuring |C(j)|^2
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
            "normalization error, value "
            + str(sum(data_b_zero + data_b_one))
            + " instead of 256"
        )
    return (data_b_zero, data_b_one)


backend = AerSimulator(
    method="matrix_product_state"
)  # method="statevector")  # method=sampler ? or default


for chunk_x in range(3):
    for chunk_y in range(3):
        pcg = pcg_k_to_l(14, 4, 17, 3)

        qc_grf = chunked_gaussian_random_field(chunk_x, chunk_y, pcg)

        j = qk.QuantumRegister(8, name="j")
        k = qk.QuantumRegister(8, name="k")
        b = qk.QuantumRegister(1, name="b")
        b1 = qk.QuantumRegister(1, name="b1")
        a_boundary = qk.QuantumRegister(2, name="ab")
        a_overflow = qk.QuantumRegister(1, name="ao")
        # dx_copy = qk.QuantumRegister(3, name="dx_c")
        # dy_copy = qk.QuantumRegister(3, name="dy_c")
        square1 = qk.QuantumRegister(14, name="sq1")
        # square2 = qk.QuantumRegister(6, name="sq2")
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

        qc_h = qc.copy()

        # qc.save_statevector()
        # measure here
        qc.measure(j, class_j)
        qc.measure(b, class_b)

        # qc.draw(output="mpl", filename="repro-circuit.png")
        print(f"Building circuit for chunk ({chunk_y}, {chunk_x})...")
        print(qc)
        circ = qk.transpile(qc, backend)

        print("Simulating circuit for magnitudes...")
        result = backend.run(circ, shots=65536).result()

        # Test if statevector is built like i think
        # cos_j_reg = result.get_statevector().data[0 : 2**8]
        # sin_j_reg = result.get_statevector().data[2**8 : 2**9]

        # print("maximal imaginary part in amplitudes of |j>|0>:")
        # print(max(cos_j_reg.imag))

        # print("Magnitude of the j register:")
        # print(sum(np.pow(np.abs(cos_j_reg), 2) + np.pow(np.abs(sin_j_reg), 2)))

        # print(result.get_counts(circ))
        prob_zero, prob_one = probabilities_from_result(result)

        shift = (prob_zero - 1 / 2) * 2
        shift = np.reshape(shift, (16, 16), copy=False)
        save_plot(shift, f"shifted-modified-chunk-{chunk_y}_{chunk_x}.png")
        save_plot(-shift, f"shifted-modified-chunk-{chunk_y}_{chunk_x}_neg.png")


# cos_data = result.get_statevector().data[0 : 2**8].real
# sin_data = result.get_statevector().data[2**8 : 2**9].real
# data.shape = (16, 16)
# data = np.reshape(cos_data - 1 / 32, (16, 16))

# save_plot(data, "quantum_circuit_repro_statevector.png")

# save_plot(np.square(np.abs(data)), "quantum_circuit_repro_statevector_abs.png")
