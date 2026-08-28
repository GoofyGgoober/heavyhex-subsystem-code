"""The 9 data qubits of the distance-3 rotated patch.

    0  1  2
    3  4  5
    6  7  8
"""

DISTANCE = 3

# One physical qubit per grid point. Index = row * 3 + col.
DATA_QUBITS = (0, 1, 2, 3, 4, 5, 6, 7, 8)


def data_qubit(row: int, col: int) -> int:
    if not (0 <= row < DISTANCE and 0 <= col < DISTANCE):
        raise ValueError(f"no data qubit at ({row}, {col})")
    return row * DISTANCE + col
