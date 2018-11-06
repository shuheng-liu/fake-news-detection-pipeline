import numpy as np
import argparse
import pickle as pkl
from scipy.sparse import csr_matrix
from doc_utils import normalized


def normalize_sparse_matrix(csr):
    return csr_matrix(np.stack(normalized(x) for x in csr.toarray()))


def load_and_dump(src, dest):
    with open(src, "rb") as fi, open(dest, "wb") as fo:
        mat = pkl.load(fi)
        pkl.dump(normalize_sparse_matrix(mat), fo)


if __name__ == '__main__':
    load_and_dump("/input/title-onehot(scorer=tfidf).pkl", "/output/title-onehot(scorer=tfidf, normalized).pkl")
    load_and_dump("/input/text-onehot(scorer=tfidf).pkl", "/output/text-onehot(scorer=tfidf, normalized).pkl")
