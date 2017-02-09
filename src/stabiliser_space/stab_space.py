import sparse_pauli as sp
import itertools as it
import numpy as np
from numpy.random import randint
from operator import or_ as union, mul
from functools import reduce
from . import gf2_mat as gf2

__all__ = ['StabSpace', 'pauli2vec']

class StabSpace(object):
    """
    Uses two lists of sparse_pauli.Pauli instances to represent
    stabiliser subspaces. 
    One list is the stabiliser generators, the other is the logical
    generators.
    """
    def __init__(self, stabs, logs=None, check=True, qubits=None, n=None):
    
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
        self.stabs = stabs
        self.logs = logs
    
        if qubits is None:
            self.qubits = sorted(list(reduce(union,
                                [p.support() for p in stabs + logs])))
        else:
            self.qubits = qubits

        if n is None:
            self.n = len(self.qubits)
        else:
            self.n = n


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
        if not(acom_stabs):
            stab_mat = np.vstack(
                [pauli2vec(s, self.qubits) for s in self.stabs]
                ).T
            try:
                aug_mat = np.hstack([stab_mat, np.matrix(pauli2vec(op, self.qubits)).T])
                decomp = gf2.solve_augmented(aug_mat)
                s = prod([a for a, b in zip(self.stabs, decomp) if b])
                return -1 if (s * op).ph else 1 # UNSAFE
            except ValueError: # inconsistent system, we assume
                # stab_log_mat = np.vstack(
                # [pauli2vec(s, self.qubits) for s in self.stabs + self.logs]
                # ).T
                # aug_mat = np.hstack([stab_log_mat, pauli2vec(op, self.qubits).T])
                # decomp = gf2.solve_augmented(aug_mat)
                meas_result = 2 * randint(2) - 1
                # eliminate the proper logicals from self.logs
                if len(self.logs) > 2:
                    raise NotImplementedError("Logical measurements in k>1 codes not supported.")
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

#---------------------------------------------------------------------#
