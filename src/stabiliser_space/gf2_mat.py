"""
Copy-paste from Ed Scheinerman's SimpleGF2.jl:
https://github.com/scheinerman/SimpleGF2.jl
"""
from numpy import int_, shape, zeros, identity, hstack
from numpy.linalg import det
from numpy.random import randint

def swap_rows(mat, r0, r1):
    temp_row = mat[r0,:].copy()
    mat[r0,:] = mat[r1,:]
    mat[r1,:] = temp_row
    pass # subroutine what a treat

def add_row_to_row(mat, r_add, r_to):
    mat[r_to, :] += mat[r_add, :]
    mat[r_to, :] %= 2
    pass # subroutine so sweet

def rref(mat):
    """
    Places a matrix into reduced-row echelon form using row-swapping
    and adding, where the addition is performed mod 2.
    """
    r, c = shape(mat)
    s = 0
    for x in range(r):
        b = False
        while not(b) and (x + s < c):
            if mat[x, x + s] == 1:
                break
            elif mat[x, x + s] == 0:
                for y in range(x, r):
                    if mat[y, x + s] == 1:
                        swap_rows(mat, y, x)
                        b = True
                        break
            if not(b):
                s = s + 1
        for m in range(r):
            if (x + s < c) and all((m != x, mat[m, x + s] == 1)):
                add_row_to_row(mat, x, m)
    pass # subroutine can't be beat

def solve_augmented(C1):
    r, c = shape(C1)
    D = C1.copy()
    rref(D)
    x = 0

    for a in range(r):
        _in = True
        for b in range(c - 1):
            if D[a, b] != 0:
                _in = False
        if _in and (D[a, -1] != 0):
            raise ValueError("Inconsistent system")

    ret = zeros(c - 1, dtype=int_)
    for p in range(r):
        if D[p, -1] == 1:
            for n in range(c - 1):
                if D[p, n] == 1:
                    ret[n] = 1
                    break
    return ret

def inv(A):
    n, m = shape(A)
    
    if n != m:
        raise ValueError("Cannot invert a matrix that isn't square.")
    
    if det(A) == 0:
        raise ValueError("Cannot invert a singular matrix.")
    
    AB = hstack([A, identity(n, int_)])
    rref(AB)

    B = AB[:, n:]
    return B

rand = lambda shp: randint(0, 2, shp)

#---------------------------------------------------------------------#
