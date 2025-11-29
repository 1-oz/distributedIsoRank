import numpy as np
from scipy.sparse import csr_matrix

def load_graph_pair(npz_path):
    dat = np.load(npz_path)

    n1 = int(dat['n1'])
    n2 = int(dat['n2'])

    A1 = csr_matrix((dat['A1_data'], dat['A1_indices'], dat['A1_indptr']), shape=(n1, n1))
    A2 = csr_matrix((dat['A2_data'], dat['A2_indices'], dat['A2_indptr']), shape=(n2, n2))
    d1 = dat['d1']
    d2 = dat['d2']

    return A1, d1, A2, d2