import ctypes as ct
import numpy as np

#load the shared object file
gf2_mat = ct.CDLL('./gf2_mat.so')

good_mat = np.array([
                    [1, 1, 1, 1],
                    [0, 1, 1, 0],
                    [0, 0, 1, 1]
                    ], dtype=np.int_)

bad_mat = np.array([
                    [1, 1, 1, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 1]
                    ], dtype=np.int_)

for mat in [good_mat, bad_mat]:
    rs, cs = np.shape(mat)
    solution = np.zeros((rs,), dtype=np.int_)
    mat_arr = np.reshape(mat, [-1])

    mat_arr = (ct.c_int * len(mat_arr))(*mat_arr)
    rs, cs = ct.c_int(rs), ct.c_int(cs)
    solution = (ct.c_int * len(solution))(*solution)

    res_int = gf2_mat.solve_augmented(mat_arr, rs, cs, solution)

    print("res_int = " + str(res_int))
    print("solution = " + str(np.array(solution)))