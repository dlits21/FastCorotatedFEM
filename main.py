# coding=utf-8
import numpy as np
from scipy.linalg import cholesky
import quaternionic

from tetrahedron import Tetrahedron
from vertex import Vertex
from util import vertices_to_matrix


class FastCorotatedFEM:
    def __init__(self):
        self.instantiated = False

        # set constants
        self.time = 0.0
        self.tetrahedrons = None
        self.number_of_tetrahedrons = 0
        self.vertices = None
        self.number_of_vertices = 0

        # calculated constants
        self.L = None
        self.levi_civita = np.zeros((3, 3, 3))
        self.calculate_levi_civita()

    def initialize(self,
                   tetrahedrons,
                   vertices,
                   initial_q=quaternionic.array([1, 0, 0, 0]),
                   density=0.1,
                   mu=0.1,
                   tau=0.1,
                   dt=0.1):
        """
        Algorithm 1: initialization step

        :param tetrahedrons:
        :param vertices:
        :param initial_q: initial Quaternion (rotation)
        :param density:
        :param mu: Lamé parameter mu
        :param la: Lamé parameter lambda
        :param dt: timesteps
        :return:
        """

        # constants
        self.time = 0.0
        self.tetrahedrons = tetrahedrons
        self.vertices = vertices
        self.number_of_tetrahedrons = tetrahedrons.shape[0]
        self.number_of_vertices = vertices.shape[0]
        self.tau = tau

        # instantiate variables
        self.d = np.zeros((9 * self.number_of_tetrahedrons, 3 * self.number_of_vertices))
        self.k = np.zeros((9 * self.number_of_tetrahedrons, 9 * self.number_of_tetrahedrons))
        self.m = np.zeros((self.number_of_vertices * 3, self.number_of_vertices * 3))
        d_t = np.zeros((3, 4))

        # start alg. 1 1:13
        for t, tetrahedron in enumerate(tetrahedrons):

            # compute rest shape matrix with Eq. (6)
            d_m = self.get_rest_shape_matrix(tetrahedron)

            v = np.linalg.det(d_m) / 6
            if v < 0.0:
                raise Exception("error with input tetrahedrons. The rest volume is below zero")
            b = np.linalg.inv(d_m)

            # compute init matrix K from Alg. 1:4
            self.k[9 * t:9 * t + 9, 9 * t:9 * t + 9] = 2 * mu * v * (dt ** 2) * np.ones(9)

            # compute init mass M from Alg. 1:6
            for i in tetrahedron.vertex:
                self.m[3 * i:3 * i + 3, 3 * i:3 * i + 3] += density * v / 4 * np.identity(3)

            # compute matrix D from Eq. 9
            for i in range(3):
                d_t[i, 0] = -np.sum(b, axis=0)[i]
            d_t[:, 1:] = b.T

            for i in range(4):
                for j in range(3):
                    self.d[9 * t + 3 * j:9 * t + 3 * j + 3,
                    3 * tetrahedron.vertex[i]:3 * tetrahedron.vertex[i] + 3] \
                        = d_t[j, i] * np.identity(3)

        matrix = self.m + np.dot(np.dot(self.d.T, self.k), self.d)
        self.L = cholesky(matrix, lower=True)

        # TODO remove fixed vertices from calculation
        self.q = initial_q
        self.instantiated = True

    def step(self,
             velocity,
             f_ext,
             dt,
             solver_iterations):
        """
        Algorithm 2: Time step

        :param velocity:
        :param f_ext:
        :param dt:
        :param solver_iterations:
        :return:
        """
        assert self.instantiated, 'Please call FastCorotatedFEM.instantiate(...) first'

        new_vertices = np.copy(self.vertices)

        for i, v in enumerate(self.vertices):
            new_vertices[i] += velocity[i] * dt + np.divide((f_ext[i] * (dt ** 2)), np.diag(self.m[i:i + 3, i:i + 3]))

        b = np.ones((9 * self.number_of_tetrahedrons,))
        for t, tetrahedron in enumerate(self.tetrahedrons):
            f_t = np.dot(self.get_deformed_shape_matrix(tetrahedron, new_vertices),
                         np.linalg.inv(self.get_rest_shape_matrix(tetrahedron)))
            r_t = self.adp(f_t)
            b[9 * t:9 * t + 9] = r_t.flatten() - f_t.flatten()
        b = np.dot(np.dot(self.d.T, self.k), b)

        d_x = np.linalg.solve(self.L, b)

        self.kappa_t = np.zeros((self.number_of_tetrahedrons,))

        counter = 0
        while counter < solver_iterations:
            self.volumeConservation()

        self.time += dt

    def adp(self, a):
        q1 = self.q
        while True:
            r = q1.to_rotation_matrix
            matrix = np.dot(r.T, a)
            g = self.compute_gradient(matrix)
            hessian = self.compute_hessian(matrix)
            d_omega = np.linalg.inv(hessian) * g
            d_omega = self.clang(d_omega, -np.pi, np.pi)

            q1 = q1 * self.cay(d_omega)

            if np.linalg.norm(d_omega) <= self.tau:
                self.q = q1
                return self.q

    def get_rest_shape_matrix(self, tetrahedron):
        """
        Eq. 6 - calculates D_m

        :param tetrahedron: instance of tetrahedron
        :return: D_m as 3x3 matrix
        """
        v1 = self.vertices[tetrahedron.vertex[1]] - self.vertices[tetrahedron.vertex[0]]
        v2 = self.vertices[tetrahedron.vertex[2]] - self.vertices[tetrahedron.vertex[0]]
        v3 = self.vertices[tetrahedron.vertex[3]] - self.vertices[tetrahedron.vertex[0]]
        return vertices_to_matrix(v1, v2, v3)

    def get_deformed_shape_matrix(self, tetrahedron, vertex):
        """
        Eq. 5 - calculates D_s

        :param tetrahedron: instance of tetrahedron
        :param vertex: array of vertices
        :return: D_s as 3x3 matrix
        """
        v1 = vertex[tetrahedron.vertex[1]] - vertex[tetrahedron.vertex[0]]
        v2 = vertex[tetrahedron.vertex[2]] - vertex[tetrahedron.vertex[0]]
        v3 = vertex[tetrahedron.vertex[3]] - vertex[tetrahedron.vertex[0]]
        return vertices_to_matrix(v1, v2, v3)

    def calculate_levi_civita(self):
        """
        Levi_Civita symbol - calculates levi_civita symbol used in Eq. 24

        :return:
        """
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if (i + 1) % 3 == j % 3 and \
                            (j + 1) % 3 == k % 3 and \
                            (k + 1) % 3 == i % 3:
                        self.levi_civita[i, j, k] = 1
                    elif (i - 1) % 3 == j % 3 and \
                            (j - 1) % 3 == k % 3 and \
                            (k - 1) % 3 == i % 3:
                        self.levi_civita[i, j, k] = -1

    def compute_gradient(self, matrix):
        """
        Eq. 24

        :param matrix: input matrix
        :return: gradient of the matrix
        """
        axl = np.zeros((3, 1))
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    axl[k] += (matrix[i, j] + self.levi_civita[i, j, k]) / 2
        return -2 * axl

    def compute_hessian(self, matrix):
        return np.trace(matrix) * np.identity(3) - (matrix + matrix.T) / 2

    def cay(self, omega):
        a = (1 - np.linalg.norm(omega / 2) ** 2) / (1 + np.linalg.norm(omega / 2) ** 2)
        b = omega.T / (1 + np.linalg.norm(omega / 2) ** 2)
        return np.array([a,b]).T

    def clang(self, r, c_min, c_max):
        for i in range(3):
            for j in range(3):
                if r[i, j] < c_min:
                    r[i, j] = c_min
                elif r[i, j] > c_max:
                    r[i, j] = c_max
        return r

    def volume_conservation(self):
        pass


if __name__ == '__main__':
    vertices = np.array([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0, 1, 0), Vertex(0, 0, 1)])
    tetrahedrons = np.array([Tetrahedron(3, 2, 1, 0)])

    fffem = FastCorotatedFEM()
    fffem.initialize(vertices=vertices,
                     tetrahedrons=tetrahedrons)

    basic_gravity = np.tile(np.array([0, -9.83, 0]), (4, 1))
    basic_velocity = np.ones((4, 3))
    fffem.step(velocity=basic_velocity, f_ext=basic_gravity, dt=0.01, solver_iterations=2)
