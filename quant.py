import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def dna_to_binary(dna_sequence):
    mapping = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
    return ''.join([mapping[nuc] for nuc in dna_sequence])


disease_dna_sequence = "ATGGATCG"
binary_sequence = dna_to_binary(disease_dna_sequence)
num_qubits = len(binary_sequence)

mutated_bit_indices = [4, 5]  # Indexing from left (0)

qc = QuantumCircuit(num_qubits, num_qubits)

for i in range(num_qubits):
    if binary_sequence[i] == '1':
        qc.x(i)
    if i in mutated_bit_indices:
        qc.barrier(i)      # Visual barrier to highlight
        qc.z(i)            # Mutation tagging (like red mark)
    qc.h(i)
qc.measure(range(num_qubits), range(num_qubits))


print("Quantum Circuit:")
print(qc.draw())


simulator = Aer.get_backend('aer_simulator')
compiled = transpile(qc, simulator)
job = simulator.run(compiled, shots=256)
result = job.result()
counts = result.get_counts()

print("\nQuantum Output (Counts):")
print(counts)

def counts_to_vector(counts, all_keys):
    total = sum(counts.values())
    return [counts.get(k, 0) / total for k in all_keys]


all_keys = sorted(counts.keys())
quantum_vector = counts_to_vector(counts, all_keys)


X_data = []
y_data = []

for _ in range(20):  # Simulate 20 samples
    qc_fake = {key: random.randint(1, 5) for key in all_keys}
    vec = counts_to_vector(qc_fake, all_keys)
    virus_count = random.randint(100, 300)
    days = random.randint(1, 7)
    location = random.choice([0, 1])  # throat=0, blood=1
    X_data.append(vec + [virus_count, days, location])
    total_load = virus_count * random.randint(200, 400)
    spread_rate = virus_count * random.randint(10, 50)
    y_data.append([total_load, spread_rate])

model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

input_sample = quantum_vector + [180, 4, 0]  
prediction = model.predict([input_sample])[0]

print("\n🧬 AI Predicted Results:")
print("Total Estimated Mutated Cell Load:", int(prediction[0]))
print("Predicted Spread Speed (per day):", int(prediction[1]))
