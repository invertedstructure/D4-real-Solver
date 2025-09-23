
import numpy as np
def to_bool(A):
    A = np.array(A)
    if A.size == 0:
        return np.zeros((0,0), dtype=bool) if A.ndim <= 1 else A.astype(bool)
    if A.ndim == 1:
        return (A.astype(np.int64) % 2).astype(bool)
    return (A.astype(np.int64) % 2).astype(bool)
def matmul_gf2(A, B):
    A = to_bool(A); B = to_bool(B)
    C = (A.astype(np.int64) @ B.astype(np.int64)) % 2
    return C.astype(bool)
def add_gf2(A, B):
    A = to_bool(A); B = to_bool(B)
    return np.logical_xor(A, B)
def eq_gf2(A, B):
    A = to_bool(A); B = to_bool(B)
    return A.shape == B.shape and np.array_equal(A, B)
def gaussian_elim_rank(A):
    A = to_bool(A).astype(np.int8)
    if A.ndim == 1:
        A = A.reshape(1,-1)
    m, n = A.shape
    R = A.copy()
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if R[r, col] & 1:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]
        for r in range(m):
            if r != row and (R[r, col] & 1):
                R[r, :] ^= R[row, :]
        rank += 1
        row += 1
        if row == m:
            break
    return rank, (R % 2).astype(bool)
def in_image(B, v):
    B = to_bool(B).astype(np.int8)
    v = to_bool(v).astype(np.int8).reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(B.shape[0], 1)
    m = B.shape[0]
    if m == 0:
        # Image is {0} in a 0-row space; only the empty/zero vector is in the image
        return v.size == 0
    if v.shape[0] != m:
        raise ValueError(f"in_image row mismatch: rows(B)={m}, len(v)={v.shape[0]}")
    n = B.shape[1]
    A = np.hstack([B, v]).astype(np.int8)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col] & 1:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        for r in range(m):
            if r != row and (A[r, col] & 1):
                A[r, :] ^= A[row, :]
        row += 1
        if row == m: break
    for r in range(m):
        if np.all(A[r, :n] % 2 == 0) and (A[r, n] % 2 != 0):
            return False
    return True
