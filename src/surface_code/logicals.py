"""Logical operators of the [[9,1,3]] rotated patch.

X-type boundaries are top and bottom, so X_L is a vertical X string.
Z-type boundaries are left and right, so Z_L is a horizontal Z string.
"""

from .pauli import Pauli
from .qubits import DISTANCE, data_qubit

# Left column. Any of the three columns is an equivalent X_L up to stabilizers.
LOGICAL_X = Pauli.x_on(data_qubit(row, 0) for row in range(DISTANCE))

# Top row. Any of the three rows is an equivalent Z_L up to stabilizers.
LOGICAL_Z = Pauli.z_on(data_qubit(0, col) for col in range(DISTANCE))
