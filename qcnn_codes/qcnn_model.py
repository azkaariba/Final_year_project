"""
Model definitions for the emotion classifier.

Both models share the exact same 5-block CNN backbone used in the earlier
Keras version (Conv -> ReLU -> BatchNorm -> MaxPool, filters 40/80/120/160/200,
64x64x3 input). They differ only in the final classification head:

  ClassicalCNN : backbone -> Dense(500) -> Dropout -> Dense(num_classes)
  QCNN         : backbone -> Linear(800, 4)  [bottleneck to 4 qubits]
                          -> 4-qubit PennyLane quantum circuit
                          -> Linear(4, num_classes)

The quantum circuit uses angle embedding (one classical feature per qubit)
followed by a small entangling variational layer, then measures each
qubit's PauliZ expectation value as the circuit's output.
"""

import math

import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4
N_QLAYERS = 2  # depth of the variational quantum circuit

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


weight_shapes = {"weights": (N_QLAYERS, N_QUBITS)}


def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2, 2),
    )


class _CNNBackbone(nn.Module):
    """Shared 5-block feature extractor - same architecture as the Keras
    version. Input: 64x64x3. Output: flattened 800-dim feature vector
    (200 channels x 2x2 spatial, after 5 halvings of 64 -> 2)."""

    def __init__(self):
        super().__init__()
        self.block1 = _conv_block(3, 40)
        self.block2 = _conv_block(40, 80)
        self.block3 = _conv_block(80, 120)
        self.block4 = _conv_block(120, 160)
        self.block5 = _conv_block(160, 200)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        return x


class ClassicalCNN(nn.Module):
    """Purely classical version of the model - backbone + Dense(500) head,
    matching the earlier Keras architecture exactly."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = _CNNBackbone()
        self.fc1 = nn.Linear(800, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class QCNN(nn.Module):
    """Hybrid quantum-classical version. Same 5-block classical backbone,
    compressed to 4 features (one per qubit), passed through a small
    quantum circuit, then a final classical layer produces class scores."""

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = _CNNBackbone()
        self.bottleneck = nn.Linear(800, N_QUBITS)
        self.quantum = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.classifier = nn.Linear(N_QUBITS, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.bottleneck(x)
        x = torch.tanh(x) * math.pi  # scale features into [-pi, pi] for angle embedding
        x = self.quantum(x)
        x = self.classifier(x)
        return x



# """
# Quantum Convolutional Neural Network (QCNN) for Handwritten Character Recognition.

# Architecture designed to achieve >80% accuracy on 62-class handwritten characters:
#     - Deep Classical CNN backbone with residual connections
#     - Quantum Variational Circuit (10 qubits, 6 layers, data re-uploading)
#     - Multi-head classical classifier with dropout

# The quantum layer operates in 2^10 = 1024 dimensional Hilbert space,
# providing exponentially richer feature representations than classical layers
# with the same parameter count.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pennylane as qml


# N_QUBITS = 10
# N_QLAYERS = 6

# dev = qml.device("default.qubit", wires=N_QUBITS)


# @qml.qnode(dev, interface="torch", diff_method="backprop")
# def quantum_circuit(inputs, weights):
#     """
#     Strongly entangling variational circuit with data re-uploading.

#     Features:
#         - Angle encoding on all qubits (RX + RY)
#         - Parametrized rotation layers (RY + RZ + RX)
#         - Full circular entanglement via CNOTs
#         - Data re-uploading every 2 layers for increased expressivity
#     """
#     for i in range(N_QUBITS):
#         qml.RX(inputs[i], wires=i)
#         qml.RY(inputs[i] * 0.5, wires=i)

#     for layer in range(N_QLAYERS):
#         for i in range(N_QUBITS):
#             qml.RY(weights[layer, i, 0], wires=i)
#             qml.RZ(weights[layer, i, 1], wires=i)
#             qml.RX(weights[layer, i, 2], wires=i)

#         for i in range(0, N_QUBITS - 1, 2):
#             qml.CNOT(wires=[i, i + 1])
#         for i in range(1, N_QUBITS - 1, 2):
#             qml.CNOT(wires=[i, i + 1])
#         qml.CNOT(wires=[N_QUBITS - 1, 0])

#         if layer % 2 == 1 and layer < N_QLAYERS - 1:
#             for i in range(N_QUBITS):
#                 qml.RY(inputs[i] * weights[layer, i, 3], wires=i)
#                 qml.RZ(inputs[i] * weights[layer, i, 4], wires=i)

#     return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# class QuantumLayer(nn.Module):
#     """Trainable quantum layer wrapping the variational circuit."""

#     def __init__(self):
#         super().__init__()
#         weight_shape = (N_QLAYERS, N_QUBITS, 5)
#         self.weights = nn.Parameter(
#             torch.nn.init.uniform_(torch.empty(weight_shape), -np.pi / 4, np.pi / 4)
#         )

#     def forward(self, x):
#         batch_size = x.shape[0]
#         outputs = []
#         for i in range(batch_size):
#             result = quantum_circuit(x[i], self.weights)
#             outputs.append(torch.stack(result))
#         return torch.stack(outputs)


# class ResBlock(nn.Module):
#     """Residual block with two conv layers and skip connection."""

#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return F.relu(out)


# class ClassicalCNNBackbone(nn.Module):
#     """
#     Deep feature extractor with residual connections.
#     Input: 1x48x48 → Output: N_QUBITS scaled features.
#     """

#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             ResBlock(64),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             ResBlock(128),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             ResBlock(256),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.reduce = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, N_QUBITS),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.reduce(x)
#         x = x * np.pi
#         return x


# class QCNN(nn.Module):
#     """
#     Hybrid Quantum-Classical CNN targeting >80% accuracy.

#     Pipeline:
#         Input (1x48x48) → Deep CNN → 10 features → Quantum Circuit (10 qubits, 6 layers)
#         → 10 expectation values → Classifier → 62 classes
#     """

#     def __init__(self, num_classes=62):
#         super().__init__()
#         self.backbone = ClassicalCNNBackbone()
#         self.quantum = QuantumLayer()
#         self.classifier = nn.Sequential(
#             nn.Linear(N_QUBITS, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, num_classes),
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         quantum_out = self.quantum(features)
#         return self.classifier(quantum_out)


# class ClassicalCNN(nn.Module):
#     """
#     Strong classical CNN baseline with equivalent depth for fair comparison.
#     Also targets >80% accuracy.
#     """

#     def __init__(self, num_classes=62):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             ResBlock(64),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             ResBlock(128),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             ResBlock(256),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.1),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         return self.classifier(x)
