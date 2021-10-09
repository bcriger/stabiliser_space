import ctypes as ct
from functools import reduce
import itertools as it
import numpy as np
from numpy.random import randint
from operator import or_ as union, mul, add
import os
import sparse_pauli as sp
# from . import gf2_mat as gf2

__all__ = ['StabSpace', 'pauli2vec', '_c_solve_augmented']

class StabSpace(object):
    """
    Uses two lists of sparse_pauli.Pauli instances to represent
    stabiliser subspaces. 
    One list is the stabiliser generators, the other is the logical
    generators.
    """
    def __init__(self, stabs=None, logs=None, check=True):
    
        if check:
            # type check
            for lst, lbl in [(stabs, 'stabiliser'), (logs, 'logical')]:
                if any([type(p) != sp.Pauli for p in lst]):
                    raise ValueError("all {}s must be sparse_pauli."
                                        "Pauli instances".format(lbl))
            # commutation check
            for pr in it.combinations(stabs, 2):
                if pr[0].com(pr[1]): # they anticommute
                    raise ValueError("Input stabilisers anticommute "
                                        ": \n{}, {}".format(*pr))
            for pr in it.product(stabs, logs):
                if pr[0].com(pr[1]): # they anticommute
                    raise ValueError("Input logical anticommutes with "
                                        "stabiliser: "
                                        "\n{}, {}".format(*pr))
        
        self.stabs = stabs if stabs else []
        self.logs = logs if logs else []

    @property
    def qubits(self):
        # TODO FIX SLOW
        return sorted(list(reduce(add, 
                [p.support() for p in self.stabs + self.logs], [])))
    
    @property
    def n(self):
        return len(self.qubits)

    # Gate Methods DON'T GET FANCY
    def cnot(self, ctrl_targs):
        for pauli in self.stabs + self.logs:
            pauli.cnot(ctrl_targs)
        pass

    def cz(self, prs):
        for pauli in self.stabs + self.logs:
            pauli.cz(prs)
        pass

    def h(self, qs):
        for pauli in self.stabs + self.logs:
            pauli.h(qs)
        pass

    def p(self, qs):
        for pauli in self.stabs + self.logs:
            pauli.p(qs)
        pass

    # Prep/Measure Methods THESE DEVIATE FROM SPARSE_PAULI
    def meas(self, op):
        """
        Here, we measure the eigenvalue of a new operator `op`, which
        we hope has eigenvalue +/- 1.
        This is Gottesman-Knill style. 
        If the new operator is a logical, we return a random bit,
        effectively assuming that the state is uniformly mixed over the
        stabiliser subspace. 
        """
        if type(op) != sp.Pauli:
            raise ValueError("op must be Pauli: "
                                "{} entered.".format(op))

        acom_stabs = [_ for _ in self.stabs if _.com(op)]
        acom_logs = [_ for _ in self.logs if _.com(op)]
        
        if not(self.stabs) and not(self.logs):
            self.stabs.append(op)
        elif not(acom_stabs):
            try:
                aug_mat = self._augmented_matrix(op)
                # decomp = gf2.solve_augmented(aug_mat)
                decomp = _c_solve_augmented(aug_mat)
                s = prod([a for a, b in zip(self.stabs, decomp) if b])
                comparison = s * op
                assert comparison.weight() == 0
                return -1 if comparison.ph else 1 # UNSAFE
            except: # inconsistent system, we assume
                # stab_log_mat = np.vstack(
                # [pauli2vec(s, self.qubits) for s in self.stabs + self.logs]
                # ).T
                # aug_mat = np.hstack([stab_log_mat, pauli2vec(op, self.qubits).T])
                # decomp = gf2.solve_augmented(aug_mat)
                meas_result = 2 * randint(2) - 1
                # eliminate the proper logicals from self.logs
                if len(self.logs) > 2:
                    raise NotImplementedError("Logical measurements in k > 1 codes not supported.")
                self.logs = []
                self.stabs.append(meas_result * op)
                return meas_result
            
        else:
            acom_p = (acom_stabs + acom_logs)[0]
            for p in acom_logs:
                self.logs.remove(p)
                if p != acom_p:
                    self.logs.append(acom_p * p)
            for p in acom_stabs:
                self.stabs.remove(p)
                if p != acom_p:
                    self.stabs.append(acom_p * p)
            meas_result = 2 * randint(2) - 1
            self.stabs.append(sp.Pauli(op.x_set, op.z_set, meas_result - 1))
            return meas_result 

    def prep(self, op):
        temp_result = self.meas(op)
        if temp_result == -1:
            for dx in range(len(self.stabs)):
                if op.__eq__(self.stabs[dx], sign=False):
                    self.stabs[dx] = op

    def _augmented_matrix(self, op):
        return np.hstack([stab_mat(self.stabs, self.qubits),
                            np.matrix(pauli2vec(op, self.qubits)).T])

    def apply(self, error):
        """
        Updates the stabiliser/logicals given a Pauli:
        """
        self.stabs = list(map(error, self.stabs))
        self.logs = list(map(error, self.logs))
        pass

    def express_stabilisers(self, stab_lst):
        """
        For book-keeping, ensures that the stabiliser generators are
        expressed using a set that we decide, rather than whatever 
        generators are calculated by meas or the like. 

        We accomplish this by writing out the desired old and new
        stabiliser sets as matrices O and N, then calculating NO^-1.
        We write out the stabilisers as products over the old
        stabilisers that correspond to unit elements in NO^-1.
        """
        old_mat = stab_mat(self.stabs, self.qubits)
        nu_stabs = []
        
        for stab in stab_lst:
            aug_mat = np.hstack([old_mat.copy(),
                                np.matrix(pauli2vec(stab, self.qubits)).T])
            # decomp = gf2.solve_augmented(aug_mat)
            decomp = _c_solve_augmented(aug_mat)
            nu_stabs.append(prod([a for a, b in zip(self.stabs, decomp) if b]))
        
        self.stabs = nu_stabs
        pass

#------------------------convenience functions------------------------#

set2vec = lambda s, bits: np.array([1 if _ in s else 0 for _ in bits])

prod = lambda itrbl: reduce(mul, itrbl, sp.Pauli())

def pauli2vec(pauli, qubits):
    """
    Takes a sparse_pauli.Pauli to a vector which is just the x and z
    vectors stuck together. 
    """
    return np.hstack([set2vec(pauli.x_set, qubits),
                        set2vec(pauli.z_set, qubits)])

def stab_mat(stabs, qubits):
    return np.vstack([pauli2vec(s, qubits) for s in stabs]).T

gf2_mat = ct.CDLL(
                    os.path.dirname(os.path.abspath(__file__)) +
                    os.path.sep +
                    'gf2_mat.so'
                )

def _c_solve_augmented(mat):
    """
    Calls out to a nearby .so file to solve an augmented linear system
    over GF(2).
    """

    mat = np.array(mat, dtype=np.int_)
    rs, cs = np.shape(mat)
    solution = np.zeros((rs,), dtype=np.int_)
    mat_arr = np.reshape(mat, [-1])

    mat_arr = (ct.c_int * len(mat_arr))(*mat_arr)
    rs, cs = ct.c_int(rs), ct.c_int(cs)
    solution = (ct.c_int * len(solution))(*solution)

    res_int = gf2_mat.solve_augmented(mat_arr, rs, cs, solution)
    
    if res_int == 1:
        raise ValueError("Inconsistent system")
    
    return solution

#---------------------------------------------------------------------#
