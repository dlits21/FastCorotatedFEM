# coding=utf-8
import numpy as np
from scipy.linalg import cholesky
import quaternionic

from tetrahedron import Tetrahedron
from vertex import Vertex
from util import vertices_to_matrix, norm


class FastCorotatedFEM:
    def __init__(self):
        """

        :var instantiated: Object has been instantiated
        :var time: current timestep
        :var tetrahedrons: numpy array of Tetrahedrons
        :var number_of_tetrahedrons: number of Tetrahedrons
        :var vertices: numpy array with Vertex
        :var number_of_vertices: number of Vertex
        :var L: Lower Triangle Matrix calculated by Cholesky factorization
        :var levi_civita: implementation of Levi-Civita symbol
        """
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
        self.__calculate_levi_civita()

    def initialize(self,
                   tetrahedrons,
                   vertices,
                   initial_q=quaternionic.array([np.pi, 0, 0, 0]),
                   density=0.1,
                   mu=0.1,
                   tau=0.1,
                   dt=0.1):
        """
        Algorithm 1: initialization step

        :param tetrahedrons: input array of tetrahedrons
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
            d_m = self.__get_rest_shape_matrix(tetrahedron)

            v = np.linalg.det(d_m) / 6
            if v < 0.0:
                raise Exception("error with input tetrahedrons. The rest volume is below zero")
            b = np.linalg.inv(d_m)

            # compute init matrix K from Alg. 1 line 1:4
            self.k[9 * t:9 * t + 9, 9 * t:9 * t + 9] = 2 * mu * v * (dt ** 2) * np.ones(9)

            # compute init mass M from Alg. 1 line 1:6
            for i in tetrahedron.vertex:
                self.m[3 * i:3 * i + 3, 3 * i:3 * i + 3] += density * v / 4 * np.identity(3)

            # compute matrix D_t from Eq. 9
            for i in range(3):
                d_t[i, 0] = -np.sum(b, axis=0)[i]
            d_t[:, 1:] = b.T

            # computes matrix D from Alg. 1 line 10:12
            for i in range(4):
                for j in range(3):
                    self.d[9 * t + 3 * j:9 * t + 3 * j + 3,
                    3 * tetrahedron.vertex[i]:3 * tetrahedron.vertex[i] + 3] \
                        = d_t[j, i] * np.identity(3)

        matrix = self.m + np.dot(np.dot(self.d.T, self.k), self.d)
        self.L = cholesky(matrix, lower=True)

        # TODO remove fixed vertices from calculation
        # initializes rotation matrix
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
            f_t = np.dot(self.__get_deformed_shape_matrix(tetrahedron, new_vertices),
                         np.linalg.inv(self.__get_rest_shape_matrix(tetrahedron)))
            r_t = self.__adp(f_t)
            b[9 * t:9 * t + 9] = r_t.flatten() - f_t.flatten()
        b = np.dot(np.dot(self.d.T, self.k), b)

        d_x = np.linalg.solve(self.L, b)

        self.kappa_t = np.zeros((self.number_of_tetrahedrons,))

        counter = 0
        while counter < solver_iterations:
            self.__volume_conservation()

        self.time += dt

    def __adp(self, a):
        """
        Algorithm 3: Analytic Polar Decomposition

        :param a: input matrix
        :return: rotational matrix
        """
        q1 = self.q

        while True:
            # get Rotation matrix Alg. 3 line 3
            r = q1.to_rotation_matrix

            # calculate gradient and hessian Alg. 3 line 4:5
            b = np.dot(r.T, a)
            gradient = self.__compute_gradient(b)
            hessian = self.__compute_hessian(b)

            # compute omega Alg. 3 line 6
            # TODO choose either to follow paper or implementation (implementation has less operations,
            #  but numpy functions generally are faster)
            # From paper
            d_omega = np.linalg.inv(hessian) * gradient
            d_omega = quaternionic.array.from_rotation_matrix(d_omega).to_axis_angle
            # From implementation
            #d_omega = self.__calc_omega(hessian, gradient)

            # clamp omega at -pi and +pi
            d_omega = self.__clang(d_omega, -np.pi, np.pi)

            # update the quaternion
            q1 = q1 * self.__cay(d_omega)

            # if norm is smaller than tau, return rotation matrix
            if norm(d_omega) <= self.tau:
                self.q = q1
                return self.q.to_rotation_matrix

    def __get_rest_shape_matrix(self, tetrahedron):
        """
        Eq. 6
        calculates D_m

        :param tetrahedron: instance of tetrahedron
        :return: D_m as 3x3 matrix
        """

        v1 = self.vertices[tetrahedron.vertex[1]] - self.vertices[tetrahedron.vertex[0]]
        v2 = self.vertices[tetrahedron.vertex[2]] - self.vertices[tetrahedron.vertex[0]]
        v3 = self.vertices[tetrahedron.vertex[3]] - self.vertices[tetrahedron.vertex[0]]

        return vertices_to_matrix(v1, v2, v3)

    def __get_deformed_shape_matrix(self, tetrahedron, vertex):
        """
        Eq. 5
        calculates D_s

        :param tetrahedron: instance of tetrahedron
        :param vertex: array of vertices
        :return: D_s as 3x3 matrix
        """

        v1 = vertex[tetrahedron.vertex[1]] - vertex[tetrahedron.vertex[0]]
        v2 = vertex[tetrahedron.vertex[2]] - vertex[tetrahedron.vertex[0]]
        v3 = vertex[tetrahedron.vertex[3]] - vertex[tetrahedron.vertex[0]]

        return vertices_to_matrix(v1, v2, v3)

    def __calculate_levi_civita(self):
        """
        Levi_Civita symbol
        calculates levi_civita symbol used in Eq. 24

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

    def __compute_gradient(self, matrix):
        """
        Eq. 24
        computes gradient of rotation matrix

        :param matrix: input rotation matrix
        :return: gradient of matrix
        """

        # TODO choose either to follow paper or implementation (implementation has less operations,
        #  but numpy functions generally are faster)
        # From Paper
        axl = np.zeros((3, 1))
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    axl[k] += (matrix[i, j] * self.levi_civita[i, j, k]) / 2

        # From implementation
        # return np.array(matrix[2,1] - matrix[1,2], matrix[0,2] - matrix[2,0], matrix[1,0] - matrix[0,1])

        return -2 * axl

    def __compute_hessian(self, matrix):
        """
        Eq. 25
        computes hessian matrix
        shorter version of:
            np.trace(matrix) * np.identity(3) - (matrix + matrix.T) / 2
        taken from implementation (see Appendix)
        advantage: uses symmetry

        :param matrix: input rotation matrix
        :return: hessian matrix
        """

        # TODO choose either to follow paper or implementation (implementation has less operations,
        #  but numpy functions generally are faster)
        # From Paper
        h = np.trace(matrix) * np.identity(3) - (matrix + matrix.T) / 2

        # From implementation
        # h = np.zeros((3, 3))
        # h[0, 0] = matrix[1, 1] + matrix[2, 2]
        # h[1, 1] = matrix[0, 0] + matrix[2, 2]
        # h[2, 2] = matrix[0, 0] + matrix[1, 1]
        # h[0, 1] = (matrix[1, 0] + matrix[0, 1]) / -2
        # h[0, 2] = (matrix[2, 0] + matrix[0, 2]) / -2
        # h[1, 2] = (matrix[2, 1] + matrix[1, 2]) / -2

        return h

    def __calc_omega(self, h, g):
        """
        Alg 3. line 6
        taken from implemetation (see Appendix)
        implementations differs from paper

        :param h: hessian matrix
        :param g: gradient
        :return: 1x3 derivative of omega
        """

        det_h = -1 * h[0, 2] * h[0, 2] * h[1, 1] + \
                2 * h[0, 1] * h[0, 2] * h[1, 2] + \
                -1 * h[0, 0] * h[1, 2] * h[1, 2] + \
                -1 * h[0, 1] * h[0, 1] * h[2, 2] + \
                1 * h[0, 0] * h[1, 1] * h[2, 2]

        omega = np.zeros((3, 1))
        factor = -1 / (4 * det_h)

        omega[0] = (h[1, 1] * h[2, 2] - h[1, 2] * h[1, 2]) * g[0] + \
                   (h[0, 2] * h[1, 2] - h[0, 1] * h[2, 2]) * g[1] + \
                   (h[0, 1] * h[1, 2] - h[0, 2] * h[1, 1]) * g[2]
        omega[0] *= factor

        omega[1] = (h[0, 2] * h[1, 2] - h[0, 1] * h[2, 2]) * g[0] + \
                   (h[0, 0] * h[2, 2] - h[0, 2] * h[0, 2]) * g[1] + \
                   (h[0, 1] * h[0, 2] - h[0, 0] * h[1, 2]) * g[2]
        omega[1] *= factor

        omega[2] = (h[0, 1] * h[1, 2] - h[0, 2] * h[1, 1]) * g[0] + \
                   (h[0, 1] * h[0, 2] - h[0, 0] * h[1, 2]) * g[1] + \
                   (h[0, 0] * h[1, 1] - h[0, 1] * h[0, 1]) * g[2]
        omega[2] *= factor

        return omega

    def __clang(self,
                r,
                c_min,
                c_max):
        """
        Alg. 3 line 7
        cuts values below c_min and above c_max
        No norm used due to implemetentation (either axis or __calc_omega(…)

        :param r: input omega
        :param c_min: lower bound
        :param c_max: upper bound
        :return: clipped/clamped rotation matrix
        """

        r[r < c_min] = c_min
        r[r > c_max] = c_max
        return r

    def __cay(self, omega):
        """
        Alg. 3 line 8
        applies a Cayley map to approximate exponential map

        :param omega: delta omega calculated in Alg. 3 line 6
        :return: quaternion of new rotation
        """

        # calculates norm
        d_omega_2 = norm(omega)

        # calculates rotation
        w = (1 - d_omega_2) / (1 + d_omega_2)

        # calculates vector
        b = omega / (1 + d_omega_2)

        return quaternionic.array([w[0], b[0][0], b[1][0], b[2][0]])

    def __volume_conservation(self):
        self.__update_x()
        self.__udpate_kappa()



if __name__ == '__main__':
    vertices = np.array([Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(0, 1, 0), Vertex(0, 0, 1)])
    tetrahedrons = np.array([Tetrahedron(3, 2, 1, 0)])
    rotation = quaternionic.array([np.pi, 0, 0, 0])

    fffem = FastCorotatedFEM()
    fffem.initialize(vertices=vertices,
                     tetrahedrons=tetrahedrons,
                     initial_q=rotation)

    basic_gravity = np.tile(np.array([0, -9.83, 0]), (4, 1))
    basic_velocity = np.ones((4, 3))
    fffem.step(velocity=basic_velocity, f_ext=basic_gravity, dt=0.01, solver_iterations=2)
