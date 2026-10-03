import math
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

class VariationalQuantumClassifier(nn.Module):
    """
    8-Qubit Variational Quantum Classifier (VQC) with Angle Embedding
    and Parameterized Quantum Layers (Rotations + Circular CNOT Entanglement).
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 1. Define PennyLane Quantum Device (State-vector simulator)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 2. Build the QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            scaled_inputs = inputs * math.pi
            qml.AngleEmbedding(scaled_inputs, wires=range(n_qubits), rotation="Y")
            
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)
                    
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                    
            return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.vqc_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.scaling = nn.Parameter(torch.tensor([2.5]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.vqc_layer(x)
        return q_out * self.scaling

def build_vqc(n_qubits: int = 8, n_layers: int = 3) -> nn.Module:
    """Returns the PyTorch-compatible Variational Quantum Classifier."""
    return VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)

if __name__ == "__main__":
    print("Testing VariationalQuantumClassifier...")
    vqc = build_vqc(n_qubits=8, n_layers=3)
    dummy_input = torch.randn(4, 8).clamp(-1.0, 1.0)
    out = vqc(dummy_input)
    print("VQC Output shape:", out.shape)
    print("Sample logits:   ", out[0].detach().numpy())
    print("VariationalQuantumClassifier test PASSED!")
