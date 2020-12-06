import numpy as np
from vertex import Vertex


def vertices_to_matrix(v1, v2, v3):
    return np.array([v1.to_array(), v2.to_array(), v3.to_array()])


def norm(v):
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
