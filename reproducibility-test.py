import qiskit as qk
import numpy as np
from qiskit_aer import AerSimulator

from random_generator import pcg6to4, constant_add
from util import save_plot
from quantum_circuit import gaussian_random_field


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


backend = AerSimulator()  # method="statevector")  # method=sampler ? or default

pcg = pcg6to4(17, 3)  # 47

qc_grf = gaussian_random_field(pcg)

j = qk.QuantumRegister(8, name="j")
b = qk.QuantumRegister(1, name="b")
b1 = qk.QuantumRegister(1, name="b1")
a_boundary = qk.QuantumRegister(2, name="ab")
a_overflow = qk.QuantumRegister(1, name="ao")
a_sq = qk.QuantumRegister(6, name="as")
a_iter = qk.QuantumRegister(2, name="ai")
class_j = qk.ClassicalRegister(8, name="j_out")
class_b = qk.ClassicalRegister(1, name="b_out")

qc = qk.QuantumCircuit(j, b, a_boundary, a_overflow, a_sq, a_iter, class_j, class_b)
qc.h(j)
qc.append(qc_grf, j[:] + b[:] + a_boundary[:] + a_overflow[:] + a_sq[:] + a_iter[:])

qc_h = qc.copy()

qc.save_statevector()
# measure here
qc.measure(j, class_j)
qc.measure(b, class_b)

# qc.draw(output="mpl", filename="repro-circuit.png")
print("Building circuit for magnitudes...")
print(qc)
circ = qk.transpile(qc, backend)

print("Simulating circuit for magnitudes...")
result = backend.run(circ, shots=10000).result()

# Test if statevector is built like i think
# cos_j_reg = result.get_statevector().data[0 : 2**8]
# sin_j_reg = result.get_statevector().data[2**8 : 2**9]

# print("maximal imaginary part in amplitudes of |j>|0>:")
# print(max(cos_j_reg.imag))

# print("Magnitude of the j register:")
# print(sum(np.pow(np.abs(cos_j_reg), 2) + np.pow(np.abs(sin_j_reg), 2)))

# print(result.get_counts(circ))
prob_zero, prob_one = probabilities_from_result(result)

qc_h.save_statevector()
qc_h.h(b)
qc_h.measure(j, class_j)
qc_h.measure(b, class_b)

print("Building circuit for Hadamard...")
print(qc_h)
circ_h = qk.transpile(qc_h, backend)
print("Simulating circuit for Hadamard...")
result_h = backend.run(circ_h, shots=10000).result()

prob_h_zero, prob_h_one = probabilities_from_result(result_h)

print(prob_h_zero + prob_h_one)


cj_with_sign = np.sign(prob_h_zero - prob_h_one) * np.sqrt(prob_zero)
just_sign = np.sign(prob_h_zero - prob_h_one)

data_counts = np.reshape(cj_with_sign, (16, 16))
save_plot(data_counts, "quantum_circuit_repro_counts.png")

print("cj parts:")
print(np.sign(result.get_statevector().data[0 : 2**8].real))

print("sqrt parts:")
print(np.sign(result.get_statevector().data[2**8 : 2**9].real))

# cheating
statevector = result.get_statevector().data
mag = np.abs(statevector[0 : 2**8].real)  # 0-255 are |j>|0> i think
sign = np.sign(
    statevector[0 : 2**8].real
    * statevector[2**8 : 2**9].real  # |j>|1> would be then in the second 256 entries
)
cheated_data = mag * sign
save_plot(np.reshape(cheated_data, (16, 16)), "cheated_data.png")


cos_data = result.get_statevector().data[0 : 2**8].real
sin_data = result.get_statevector().data[2**8 : 2**9].real
# data.shape = (16, 16)
data = np.reshape(cos_data, (16, 16))

# Data analysis:
largest_angle = 0.0
for i in range(2**8):
    if np.sign(cos_data[i]) != np.sign(prob_h_zero[i] - prob_h_one[i]):
        angle = np.arcsin(sin_data[i])
        print(
            "Sign Difference: i "
            + str(i)
            + ", cosine: "
            + str(cos_data[i])
            + ", from sin: "
            + str(angle)
        )
        if abs(angle) > largest_angle:
            largest_angle = abs(angle)

print(f"largest (absolute) angle: {largest_angle}")

save_plot(data, "quantum_circuit_repro_statevector.png")

save_plot(np.square(np.abs(data)), "quantum_circuit_repro_statevector_abs.png")
