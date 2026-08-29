"""The 8 stabilizer tiles on the 3x3 data patch.

    0  1  2
    3  4  5
    6  7  8

Z-checks catch X errors. X-checks catch Z errors.
"""

from .pauli import Pauli
from .qubits import data_qubit

# 4 Z tiles: two fat interior squares, two skinny boundary pairs.
Z_CHECKS = (
    frozenset({data_qubit(0, 0), data_qubit(0, 1), data_qubit(1, 0), data_qubit(1, 1)}),  # 0,1,3,4
    frozenset({data_qubit(1, 1), data_qubit(1, 2), data_qubit(2, 1), data_qubit(2, 2)}),  # 4,5,7,8
    frozenset({data_qubit(1, 0), data_qubit(2, 0)}),  # left  3,6
    frozenset({data_qubit(0, 2), data_qubit(1, 2)}),  # right 2,5
)

# 4 X tiles: the other two interior squares, plus top and bottom.
X_CHECKS = (
    frozenset({data_qubit(0, 1), data_qubit(0, 2), data_qubit(1, 1), data_qubit(1, 2)}),  # 1,2,4,5
    frozenset({data_qubit(1, 0), data_qubit(1, 1), data_qubit(2, 0), data_qubit(2, 1)}),  # 3,4,6,7
    frozenset({data_qubit(0, 0), data_qubit(0, 1)}),  # top    0,1
    frozenset({data_qubit(2, 1), data_qubit(2, 2)}),  # bottom 7,8
)

Z_STABILIZERS = tuple(Pauli.z_on(support) for support in Z_CHECKS)
X_STABILIZERS = tuple(Pauli.x_on(support) for support in X_CHECKS)
STABILIZERS = X_STABILIZERS + Z_STABILIZERS
