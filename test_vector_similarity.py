# encoding=utf-8

import numpy as np

from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def test_cos_sim():
    """
    :return:
    """
    a = np.array([[1, 2, 3]])
    b = np.array([[2, 4, 6]])
    c = np.array([[3, 6, 9]])

    sim_ab = cosine_similarity(a, b)
    sim_ac = cosine_similarity(a, c)

    sim_ab = 1 - cosine(a, b)  # 1 - cos_distance = cos similarity
    sim_ac = 1 - cosine(a, c)

    sim_ab = np.squeeze(sim_ab)
    sim_ac = np.squeeze(sim_ac)

    print("sim_ab: {:.5f}".format(sim_ab))
    print("sim_ac: {:.5f}".format(sim_ac))


if __name__ == "__main__":
    test_cos_sim()
