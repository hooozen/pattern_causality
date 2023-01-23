import numpy as np
from functools import reduce
from scipy import stats as st


def erf(x):
    return 2 * st.norm.cdf(2**(1/2)*x) - 1


class PC:
    def __init__(self, X, Y, E=2, tau=1, p=2, h=0, nn_no=None, start=None, end=None) -> None:
        self.X = X
        self.Y = Y
        self.E = E
        self.tau = tau
        self.p = p
        self.h = h

        if start is None:
            start = 1 + E + (E-1) * tau
        if end is None:
            end = X.size - (E-1) * tau
        if nn_no is None:
            nn_no = E + 1

        self.start = start
        self.end = end
        self.L = end - start
        self.nn_no = nn_no

        self.MX = None
        self.MY = None

        self.MX_dist = None
        self.MY_dist = None

        self.nn_d_MX = None
        self.nn_d_MY = None
        self.nn_ind_MX = None
        self.nn_ind_MY = None

        self.MX_p = None
        self.MY_p = None

    def cacl(self, save_to=None):
        p = self.p
        h = self.h

        self.MX = self._retrieve_shadow_attractor(self.X)
        self.MY = self._retrieve_shadow_attractor(self.Y)

        self.MX_dist = self._get_dist(self.MX, p)
        self.MY_dist = self._get_dist(self.MY, p)

        self.nn_d_MX, self.nn_ind_MX = self._get_nn_dist(self.MX_dist)
        self.nn_d_MY, self.nn_ind_MY = self._get_nn_dist(self.MY_dist)

        self.MX_p = self._calc_pattern(self.MX)
        self.MY_p = self._calc_pattern(self.MY)

        self.MY_p_hat = self._predict_pattern(
            self.nn_ind_MX, self.MY_p, self.MY_dist)
        # self.MX_p_hat = self._predict_pattern(self.nn_ind_MY, self.MX_p, self.MX_dist)

        self.pm_XY = self._get_pattern_matrixs(
            self.MX_p, self.MY_p, self.MY_p_hat)
        # self.pm_YX = self._get_pattern_matrixs(self.MY_p, self.MX_p, self.MX_p_hat)

        self.casualities_XY = self._get_casualities(self.pm_XY)
        # self.casualities_YX = self._get_casualities(self.pm_YX)

        if save_to is None:
            np.savetxt('result.csv', self.casualities_XY, delimiter=',')

        return self.casualities_XY

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
            # mx[E - i - 1] = X[i * tau: i * tau + l]
            mx[i] = X[i * tau: i * tau + l]

        return mx.T

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
        for i in range(1, E):
            values[:, i-1] = (mx[:, i] - mx[:, i-1]) / mx[:, i-1]

        symbol[values < 0] = -1
        symbol[values > 0] = 1

        return values, symbol

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

    def _get_nn_dist(self, dist, nn_ind=None):
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
        start = self.start
        end = self.end

        nn_dist = np.full((dist.shape[0], no), np.nan)

        if nn_ind is None:

            nn_ind = np.full((nn_dist.shape[0], no), -1, dtype=np.int_)

            for i in range(start, end):
                # line = dist[i, :i]
                line = dist[i]
                nn_dist[i] = np.sort(line)[:no]
                nn_ind[i] = np.argsort(line)[:no]

            return nn_dist, nn_ind

        h = self.h
        nn_ind = (nn_ind + h)[start: nn_ind.shape[0]-h]
        nn_ind = np.vstack([np.full((h + start, no), -1), nn_ind])

        nn_dist[start+h:] = dist[np.arange(start+h, end)[:, None], nn_ind]
        '''
        for i in range(start+h, end):
            nn_dist[i] = dist[i, nn_ind[i]]
        '''

        return nn_dist, nn_ind

    def _get_dist_w(self, nn_d):
        ''' get distance weight according `nn_d`
        Params:
            nn_d: matrix(l * no) distances of nearest neighborhoods
        Returns: matrix(l * no) weights of every distance
        '''
        nn_w = nn_d / np.sum(nn_d, axis=1).reshape((-1, 1))
        return np.exp(-nn_w) / np.sum(np.exp(-nn_w), axis=1).reshape((-1, 1))

    def _get_average_signature(self, patterns, weights):
        ''' calculate average signatrue of patterns using giving weights
        Params:
            patterns: matrix(l * no * E-1), pattern values of each row vectros' nearest neighborhoods
            weights: matrix(l * no * E-1), weights of each pattern values
        Returns:
            p_values: matrix(l, E - 1), average patterns values at every time in patterns
            p_signature: matrix(l, E - 1), symbol of p_values
        '''
        l, no, _ = patterns.shape
        E = self.E
        p_values = np.zeros((l, E-1))
        for i in range(l):
            if np.count_nonzero(patterns[i]) <= no + 1 - E:
                p_values[i] = 0
            else:
                p_values[i] = sum(weights[i] @ patterns[i])
        p_signature = np.zeros(p_values.shape)

        p_signature[p_values > 0] = 1
        p_signature[p_values < 0] = -1

        return p_values, p_signature

    def _get_pattern_matrixs(self, x_p, y_p, y_p_hat):
        start = self.start
        end = self.end
        E = self.E
        L = self.L
        h = self.h
        p = self.p
        x_v, x_s = [p[start: end - h] for p in x_p]
        y_v, y_s = [p[start + h: end] for p in y_p]
        y_hat_v, y_hat_s = [p[start + h: end] for p in y_p_hat]

        pattern_matrixs = np.empty((L - h, 3**(E-1), 3**(E-1)))
        # pattern_matrixs.fill(-1)
        pattern_matrixs.fill(0)

        hit_ind = np.logical_and.reduce((y_s == y_hat_s).T)
        hit_y_v = y_v[hit_ind]
        hit_y_v_hat = y_hat_v[hit_ind]
        hit_y_s = y_s[hit_ind]
        hit_x_v = x_v[hit_ind]
        hit_x_s = x_s[hit_ind]

        j = 0
        for i in range(len(hit_ind)):
            if not hit_ind[i]:
                continue

            m_i = self._get_symbolic_ind(hit_x_s[j])
            m_j = self._get_symbolic_ind(hit_y_s[j])
            x_v = hit_x_v[j]
            y_hat_v = hit_y_v_hat[j]
            # y_hat_v = hit_y_v[j]
            pattern_matrixs[i, m_i, m_j] = erf(
                np.linalg.norm(y_hat_v, ord=p) / np.linalg.norm(x_v, ord=p))

            j += 1

        return pattern_matrixs

    def _predict_pattern(self, x_nn_ind, y_p, y_dist):
        nn_hat_d, _ = self._get_nn_dist(y_dist, x_nn_ind)
        nn_hat_w = self._get_dist_w(nn_hat_d)
        nn_hat_pattern_values = y_p[0][x_nn_ind]

        pattern_hat = self._get_average_signature(
            nn_hat_pattern_values, nn_hat_w)

        return pattern_hat

    def _get_casuality(self, pattern_matrix):
        n, _ = pattern_matrix.shape
        casuality = np.zeros(3)
        for i in range(n):
            for j in range(n):
                c = pattern_matrix[i, j]
                if c == -1:
                    continue
                if i == j:
                    casuality[0] += c
                elif i + j == n - 1:
                    casuality[1] += c
                else:
                    casuality[2] += c
        return casuality

    def _get_casualities(self, pattern_matrixs):
        l = pattern_matrixs.shape[0]
        casualities = np.empty((l, 3))
        for i in range(l):
            casualities[i] = self._get_casuality(pattern_matrixs[i])

        return casualities

    def _get_symbolic_ind(self, x):
        return int(reduce(lambda x, y: x + 3 ** y[0] * (y[1] + 1),
                          enumerate(x), 0))
