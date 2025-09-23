
import numpy as np

def to_bool(A):
    A = np.array(A)
    if A.ndim == 0:
        return np.array([[bool(A % 2)]])
    return (A.astype(np.int8) % 2).astype(bool)

def kron_gf2(A, B):
    A = to_bool(A).astype(np.int8); B = to_bool(B).astype(np.int8)
    return (np.kron(A, B) % 2).astype(np.int8)

def vec_gf2(M):
    M = to_bool(M).astype(np.int8)
    return M.reshape(-1, 1)

def unvec_gf2(v, rows, cols):
    v = (np.array(v).astype(np.int8) % 2).reshape(rows*cols, 1)
    return v.reshape(rows, cols)

def solve_gf2(A, b):
    """Solve A x = b over GF(2) (least one solution).
    Returns one solution x (n x 1) if consistent, else raises ValueError.
    """
    A = (np.array(A).astype(np.int8) % 2)
    b = (np.array(b).astype(np.int8) % 2).reshape(-1,1)
    m, n = A.shape
    # Augment
    M = np.hstack([A, b]).astype(np.int8)
    row = 0
    pivots = []
    for col in range(n):
        # find pivot
        pivot = None
        for r in range(row, m):
            if M[r, col] & 1:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        # eliminate other rows
        for r in range(m):
            if r != row and (M[r, col] & 1):
                M[r, :] ^= M[row, :]
        pivots.append(col)
        row += 1
        if row == m: break
    # Check consistency (row of zeros with RHS 1)
    for r in range(row, m):
        if (M[r, :n] % 2 == 0).all() and (M[r, n] & 1):
            raise ValueError("Inconsistent system over GF(2).")
    # Back-substitute for a basic solution
    x = np.zeros((n,1), dtype=np.int8)
    # Work backwards
    for r in range(min(row, len(pivots))-1, -1, -1):
        c = pivots[r]
        # sum over known vars
        s = 0
        for j in range(c+1, n):
            if M[r, j] & 1:
                s ^= int(x[j,0])
        x[c,0] = (M[r, n] ^ s) & 1
    return x % 2
