import numpy as np
import matplotlib.pyplot as plt
import scipy

def l(k: int, index: int):
    return mu * x[index, k] * (1 - x[index, k])

def l_prime(k: int, index: int):
    return mu * (1 - 2 * x[index, k])

def simulate(k: int):
    vy = np.random.uniform(low=0, high=a)
    vx = np.random.uniform(low=0, high=a)
    x[0, k] = (1 - alpha) * l(k - 1, 0) + (1 - vy) * alpha * l(k - 1, 1)
    x[1, k] = (1 - alpha) * l(k - 1, 1) + (1 - vx) * alpha * l(k - 1, 0)

    wx = np.random.uniform(low=0, high=b)
    wy = np.random.uniform(low=0, high=b)
    z[:, k] = x[:2, k] * (1 - np.array([wx, wy]))
    pass

class EKF:
    def __init__(self, T, x0, P0) -> None:
        self.xp = np.zeros((n_states, T))
        self.xm = np.zeros((n_states, T))
        self.xm[:, 0] = x0
        self.Pp = np.zeros((T, n_states, n_states))
        self.Pm = np.zeros((T, n_states, n_states))
        self.Pm[0] = P0
        pass

    def _l(self, k: int, index: int):
        return mu * self.xm[index, k] * (1 - self.xm[index, k])

    def _l_prime(self, k: int, index: int):
        return mu * (1 - 2 * self.xm[index, k])

    def _h(self, k: int):
        return self.xp[:2, k] * (1 - np.array([b/2, b/2]))
    
    def _prior_update(self, k: int):

        if PART == "a":
            A = np.array([
                [(1 - alpha) * self._l_prime(k-1, 0), (1 - a/2) * alpha * self._l_prime(k-1, 1)],
                [(1 - a/2) * alpha * self._l_prime(k-1, 0), (1 - alpha) * self._l_prime(k-1, 1)]
            ])
            L = np.array([
                [0, -alpha * self._l(k-1, 1)],
                [-alpha * self._l(k-1, 0), 0]
            ])
            self.xp[:, k] = np.array([
                (1 - alpha) * self._l(k-1, 0) + (1 - a/2) * alpha * self._l(k-1, 1),
                (1 - alpha) * self._l(k-1, 1) + (1 - a/2) * alpha * self._l(k-1, 0)
            ])
        else:
            A = np.array([
                [(1 - self.xm[2, k-1]) * self._l_prime(k-1, 0), (1 - a/2) * self.xm[2, k-1] * self._l_prime(k-1, 1), -self._l(k-1, 0) + (1 - a/2) * self._l(k-1, 1)],
                [(1 - a/2) * self.xm[2, k-1] * self._l_prime(k-1, 0), (1 - self.xm[2, k-1]) * self._l_prime(k-1, 1), -self._l(k-1, 1) + (1 - a/2) * self._l(k-1, 0)],
                [0, 0, 1]
            ])
            L = np.array([
                [0, -self.xm[2, k-1] * self._l(k-1, 1)],
                [-self.xm[2, k-1] * self._l(k-1, 0), 0],
                [0, 0]
            ])
            self.xp[:, k] = np.array([
                (1 - self.xm[2, k-1]) * self._l(k-1, 0) + (1 - a/2) * self.xm[2, k-1] * self._l(k-1, 1),
                (1 - self.xm[2, k-1]) * self._l(k-1, 1) + (1 - a/2) * self.xm[2, k-1] * self._l(k-1, 0),
                self.xm[2, k-1]
            ])

        self.Pp[k, :, :] = A @ self.Pm[k-1, :, :] @ A.T + L @ Q @ L.T
        pass

    def _posterior_update(self, k: int, z_k):
        if PART == "a":
            H = np.array([
                [1 - b/2, 0],
                [0, 1 - b/2]
            ])
        else:
            H = np.array([
                [1 - b/2, 0, 0],
                [0, 1 - b/2, 0],
            ])

        M = np.array([
            [-self.xp[0, k], 0],
            [0, -self.xp[1, k]]
        ])
        K = self.Pp[k] @ H.T @ np.linalg.inv(H @ self.Pp[k] @ H.T + M @ R @ M.T)

        self.xm[:, k] = self.xp[:, k] + K @ (z_k - self._h(k))
        self.Pm[k, :, :] = (np.eye(n_states) - K @ H) @ self.Pp[k]
        pass

    def update(self, k: int, z_k):
        self._prior_update(k)
        self._posterior_update(k , z_k)

if __name__ == "__main__":
    PART = input("Choose part of the problem to run: ")

    # constants
    T = 200
    mu = 3.9
    a = 0.6
    b = 0.15
    sensors = 2
    Q = a**2/12 * np.eye(2)
    R = b**2/3 * np.eye(2)

    # init
    if PART == "a":
        n_states = 2
        alpha = 0.07
        x0 = np.array([0.5, 0.5])
        P0 = 1/12 * np.eye(n_states)
    else:
        n_states = 3
        alpha = np.random.uniform(0, 0.1)
        x0 = np.array([0.5, 0.5, 1/20])
        P0 = 1/12 * np.eye(n_states)
        P0[2, 2] = 1/1200

    x = np.zeros((n_states, T))
    x[:, 0] = np.random.rand(n_states)
    if PART == "b":
        x[2, :] = alpha

    z = np.zeros((sensors, T))

    ekf = EKF(T, x0, P0)

    # simulate
    for k in range(1, T):
        simulate(k)
        ekf.update(k, z[:, k])
    
    if PART == "a":
        fig, axs = plt.subplots(2, 1, layout='constrained')
        axs[0].plot(ekf.xm[0, :], label="X_hat")
        axs[0].plot(x[0, :], label="X true")
        axs[0].legend()

        axs[1].plot(ekf.xm[1, :], label="Y_hat")
        axs[1].plot(x[1, :], label="Y true")
        axs[1].legend()

        plt.show()
    else:
        fig, axs = plt.subplots(3, 1, layout='constrained')
        axs[0].plot(ekf.xm[0, :], label="X_hat")
        axs[0].plot(x[0, :], label="X true")
        axs[0].legend()

        axs[1].plot(ekf.xm[1, :], label="Y_hat")
        axs[1].plot(x[1, :], label="Y true")
        axs[1].legend()

        axs[2].plot(ekf.xm[2, :], label="alpha_hat")
        axs[2].plot(x[2, :], label="alpha true")
        axs[2].legend()

        plt.show()
    pass