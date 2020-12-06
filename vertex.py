import numpy as np


class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        if isinstance(other, np.ndarray):
            x = self.x - other[0]
            y = self.y - other[1]
            z = self.z - other[2]
        elif isinstance(other, object):
            x = self.x - other.x
            y = self.y - other.y
            z = self.z - other.z
        else:
            x = self.x - other
            y = self.y - other
            z = self.z - other
        return Vertex(x, y, z)

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            x = self.x + other[0]
            y = self.y + other[1]
            z = self.z + other[2]
        elif isinstance(other, object):
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
        else:
            x = self.x + other
            y = self.y + other
            z = self.z + other
        return Vertex(x, y, z)

    def __radd__(self, other):
        if isinstance(other, np.ndarray):
            x = self.x + other[0]
            y = self.y + other[1]
            z = self.z + other[2]
        elif isinstance(other, object):
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
        else:
            x = self.x + other
            y = self.y + other
            z = self.z + other
        return Vertex(x, y, z)

    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            x = self.x + other[0]
            y = self.y + other[1]
            z = self.z + other[2]
        elif isinstance(other, object):
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
        else:
            x = self.x + other
            y = self.y + other
            z = self.z + other
        return Vertex(x, y, z)

    def __mul__(self, other):
        if type(other) == 'type':
            x = self.x * other.x
            y = self.y * other.y
            z = self.z * other.z
        elif isinstance(other, np.ndarray):
            x = self.x * other[0]
            y = self.y * other[1]
            z = self.z * other[2]
        else:
            x = self.x * other
            y = self.y * other
            z = self.z * other
        return Vertex(x, y, z)

    def __rmul__(self, other):
        if type(other) == 'type':
            x = self.x + other.x
            y = self.y + other.y
            z = self.z + other.z
        else:
            x = self.x + other
            y = self.y + other
            z = self.z + other
        return Vertex(x, y, z)

    def to_array(self):
        return np.array([self.x, self.y, self.z])

    def update_gravity(self, dt):
        self.y -= dt * 9.81