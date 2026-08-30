"""One ancilla per stabilizer tile.

Data stay 0..8. Ancillas are 9..16, in the same order as STABILIZERS
(X tiles first, then Z tiles).
"""

X_ANCILLAS = (9, 10, 11, 12)
Z_ANCILLAS = (13, 14, 15, 16)
ANCILLAS = X_ANCILLAS + Z_ANCILLAS
