import numpy as np


def cosine_similarity(e1, e2):
    e1 = e1.flatten()
    e2 = e2.flatten()
    return np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
