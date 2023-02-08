import numpy as np
from functools import reduce
from scipy import stats as st
from time import time
import os


def erf(x):
    return 2 * st.norm.cdf(2 ** (1 / 2) * x) - 1


class PC:
    def __init__(self, X, Y, lib_size, E=2, tau=1, p=2, nn_no=None) -> None:
        self.X = X
        self.Y = Y
        self.lib_size = lib_size
        self.E = E
        self.tau = tau
        self.p = p
        self.end = X.size - (E - 1) * tau
        self.start = abs(tau) * (E - 1)

        self.L = self.end
        if nn_no is None:
            nn_no = E + 1
        self.nn_no = nn_no

        self.MX = self._retrieve_shadow_attractor(
            self.X)  # shadow attractor of X, L * E
        self.MY = self._retrieve_shadow_attractor(
            self.Y)  # shadow attractor of Y, L * E

        # distance matrix of MX, L * L
        self.MX_dist = self._get_dist(self.MX, p)
        # distance matrix of MY, L * L
        self.MY_dist = self._get_dist(self.MY, p)

        # N nearest neighbors of MX (distance, index), 2 * L * nn_no,
        self.MX_nearest_neighbors = None
        # N nearest neighbors of MX (distance, index), 2 * L * nn_no
        self.MY_nearest_neighbors = None

        # pattern of MX (values, token), 2 * L * (E - 1)
        self.MX_pattern = None
        # pattern of MY (values, token), 2 * L * (E - 1)
        self.MY_pattern = None

        self.MX_signature = None  # signature of MX, L * (E - 1)
        self.MY_signature = None  # signature of MY, L * (E - 1)

    def ccm(self):
        sub_lib = np.arange(*self.lib_size)
        rho_y2x = np.zeros(sub_lib.size)
        rho_x2y = np.zeros(sub_lib.size)
        j = 0
        for i in sub_lib:

            mx_knn_neighbors = self._get_nearest_neighbors(
                self.MX_dist[:, :i])
            my_knn_neighbors = self._get_nearest_neighbors(
                self.MY_dist[:, :i])

            Y_hat = self._ccm_predict(self.Y, mx_knn_neighbors)
            X_hat = self._ccm_predict(self.X, my_knn_neighbors)

            rho_x2y[j] = st.pearsonr(self.Y[self.start:], Y_hat).statistic
            rho_y2x[j] = st.pearsonr(self.X[self.start:], X_hat).statistic
            j += 1

        self.ccm = np.vstack((rho_x2y, rho_y2x))
        return self.ccm

    def pc(self, save_to=None, log=None):

        self.MX_pattern = self._calc_pattern(self.MX)
        self.MY_pattern = self._calc_pattern(self.MY)

        self.MX_nearest_neighbors = self._get_nearest_neighbors(self.MX_dist)
        self.MY_nearest_neighbors = self._get_nearest_neighbors(self.MY_dist)

        self.MX_signature = self._get_signature(
            self.MX_pattern[0], self.MX_nearest_neighbors)
        self.MY_signature = self._get_signature(
            self.MY_pattern[0], self.MY_nearest_neighbors)

        sub_lib = np.arange(*self.lib_size)
        casuality = np.zeros((2, sub_lib.size, 3))
        j = 0
        for i in sub_lib:

            mx_knn_neighbors = self._get_nearest_neighbors(
                self.MX_dist[:, :i])
            my_knn_neighbors = self._get_nearest_neighbors(
                self.MY_dist[:, :i])

            my_signature_hat = self._get_signature(
                self.MY_pattern[0], mx_knn_neighbors)

            mx_signature_hat = self._get_signature(
                self.MX_pattern[0], my_knn_neighbors)

            y2x_matrix = self._get_pattern_matrixs(
                self.MX_signature, self.MY_signature, my_signature_hat)
            x2y_matrix = self._get_pattern_matrixs(
                self.MY_signature, self.MX_signature, mx_signature_hat)

            casualities_x2y = self._get_casuality(x2y_matrix[0])
            casualities_y2x = self._get_casuality(y2x_matrix[0])

            casuality[0, j, :] = casualities_x2y
            casuality[1, j, :] = casualities_y2x

            j += 1

        return casuality

    def _retrieve_shadow_attractor(self, X):
        ''' retrieve shadow attractor
        Params:
            X: list(L), time series
        Returns: matrix(L-(E-1)tau * E), shadow attractor
        '''
        E = self.E
        tau = self.tau

        l = X.shape[0] - (E - 1) * tau
        mx = np.empty((E, l))
        for i in range(E):
            mx[E - i - 1] = X[i * tau: i * tau + l]
            # mx[i] = X[i * tau: i * tau + l]

        return mx.T

    def _get_dist(self, mx, p=2):
        '''get dist matrix of mx
        Params:
            mx: matrix(l * E), shadow attractor
            p: number, L_p distance
        Returns: matrix(l * l), distance matrix of all row verctor in mx
        '''
        l, E = mx.shape
        dist = np.empty((l, l))
        dist.fill(float('inf'))
        for i in range(l):
            for j in range(i + 1, l):
                dist[i, j] = np.linalg.norm(mx[i] - mx[j], ord=p)
                dist[j, i] = dist[i, j]

        return dist

    def _get_nearest_neighbors(self, dist):
        ''' get nearest neighborhoods of every row verctor in distance matrix.
            if nn_ind = None, calculate the top `no` nearest neightborhoods,
            else return verctors in `dist` with indexes `nn_ind`
        Params:
            dist: matrix(l * l), distance matrix
            nn_ind: matrix(l * no), indexes of `no` distance matrix row verctors for every row
        Returns:
            nn_dist: matrix(l * no)
            nn_ind: matrix(l * no) indexes of nn_dist
        '''
        no = self.nn_no
        L = dist.shape[0]

        n_neighbors_dist = np.full((L, no), np.nan)
        n_neighbors_ind = np.full((L, no), np.nan, dtype=int)

        n_neighbors_ind = np.argsort(dist, axis=1)[:, :no]
        n_neighbors_dist = dist[np.arange(L)[:, None], n_neighbors_ind]

        return n_neighbors_dist, n_neighbors_ind

    def _get_neighbors_weight(self, nn_dist):
        ''' get distance weight according `nn_d`
        Params:
            nn_d: matrix(l * no) distances of nearest neighborhoods
        Returns: matrix(l * no) weights of every distance
        '''
        min_weight = 1e-6

        zero_row_mask = nn_dist[:, 0] == 0
        zero_row = nn_dist[zero_row_mask]
        other_row = nn_dist[~zero_row_mask]
        zero_weight_mask = zero_row > 0
        zero_row[zero_weight_mask] = 0
        zero_row[~zero_weight_mask] = 1
        other_row = np.exp(-other_row / other_row[:, 0].reshape(-1, 1))

        weights = np.empty_like(nn_dist)
        weights[zero_row_mask] = zero_row
        weights[~zero_row_mask] = other_row
        weights /= weights.sum(axis=1).reshape(-1, 1)
        weights[weights < min_weight] = min_weight
        return weights

    def _ccm_predict(self, x, neighbors):
        nn_dist = neighbors[0]
        nn_ind = neighbors[1] + self.start
        weight = self._get_neighbors_weight(nn_dist)
        x_hat = np.multiply(x[nn_ind], weight)
        return x_hat.sum(axis=1)

    def _calc_pattern(self, mx):
        ''' calculate pattern shadow attractor
        Params:
            mx: matrix(l * E), shadow attractor
        Returns:
            values: matrix(l * E-1), successive percentage changes of mx
            symbol: matrix(l * E-1), symbol of `values` with -1, 1 and 0
        '''
        l, E = mx.shape
        values = np.empty((l, E-1))
        symbol = np.zeros(values.shape)
        for i in range(0, E - 1):
            values[:, i] = (mx[:, i] - mx[:, i + 1]) / mx[:, i + 1]

        symbol[values < 0] = -1
        symbol[values > 0] = 1

        return values, symbol

    def _get_signature(self, patterns, nearest_neighbors):
        E = self.E
        l = self.L

        p_values = patterns[nearest_neighbors[1]]
        weights = self._get_neighbors_weight(nearest_neighbors[0])

        average_pattern = np.zeros((l, E - 1))
        signature = np.zeros(average_pattern.shape)
        '''
        for i in range(l):
            if np.count_nonzero(p_values[i]) <= no + 1 - E:
                average_pattern[i] = 0
            else:
                average_pattern[i] = weights[i] @ p_values[i]

        '''
        average_pattern = np.multiply(p_values, np.repeat(
            weights[:, :, None], self.E-1, axis=2)).sum(axis=1)
        signature[average_pattern > 0] = 1
        signature[average_pattern < 0] = -1

        return signature, average_pattern

    def _get_pattern_matrixs(self, signature_e, signature_c, signature_c_hat):
        E = self.E
        hit_cnt = np.zeros((3 ** (E - 1), 3 ** (E - 1)))
        strength = np.zeros(hit_cnt.shape)

        hit_ind = np.logical_and.reduce(
            (signature_c[0] == signature_c_hat[0]), axis=1)

        for i in range(len(hit_ind)):
            if not hit_ind[i]:
                continue

            m_i = self._get_symbolic_ind(signature_c[0][i])
            m_j = self._get_symbolic_ind(signature_e[0][i])
            hit_cnt[m_i, m_j] += 1
            strength[m_i, m_j] += erf(
                np.linalg.norm(signature_e[1][i], ord=self.p) / np.linalg.norm(signature_c[1][i], ord=self.p))

        _hit_cnt = np.copy(hit_cnt)
        _hit_cnt[hit_cnt == 0] = 1
        return hit_cnt, strength / _hit_cnt

    def _get_casuality(self, matrix):
        mid = int(matrix.shape[0] / 2)
        casuality = np.zeros(3)
        casuality[0] = matrix.trace() - matrix[mid, mid]
        casuality[1] = matrix[:, ::-1].trace() - matrix[mid, mid]
        casuality[2] = matrix.sum() - casuality[0] - casuality[1]
        return casuality

    def _get_symbolic_ind(self, x):
        return int(reduce(lambda x, y: x + 3 ** y[0] * (y[1] + 1),
                          enumerate(x), 0))
