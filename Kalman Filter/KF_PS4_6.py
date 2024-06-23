import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import scipy.linalg

def check_stability():
    A_eigen_values, _ = np.linalg.eig(A)

    result = np.sum(np.where(np.abs(A_eigen_values) < 1, 0, 1))
    return "Yes!" if result == 0 else f"No, there are {result} poles that are greater than 1."

def check_observability():
    O = np.zeros((H.shape[0]*A.shape[0], A.shape[0]))
    temp_A = np.eye(A.shape[0], A.shape[1])
    for i in range(H.shape[1]):
        O[i*H.shape[0]:(i+1)*H.shape[0], :] = H @ temp_A
        temp_A = temp_A @ A
    rank = np.linalg.matrix_rank(O)

    return "Yes!" if rank == H.shape[1] else "No."

def simulate(k: int):
    v = np.random.multivariate_normal(mean=np.zeros((16,)), cov=Q)
    w = np.random.multivariate_normal(mean=np.zeros((18,)), cov=R)

    x[:, k] = A @ x[:, k-1] + v
    z[:, k] = H @ x[:, k] + w
    pass

class KF:
    def __init__(self, T, x0, P0) -> None:
        self.xp = np.zeros((x0.shape[0], T))
        self.xm = np.zeros((x0.shape[0], T))
        self.Pp = np.zeros((T, P0.shape[0], P0.shape[1]))
        self.Pm = np.zeros((T, P0.shape[0], P0.shape[1]))

        # init
        self.xm[:, 0] = x0
        self.Pm[0, :] = P0
        pass

    def _prior_update(self, k: int):
        self.xp[:, k] = A @ self.xm[:, k-1]
        self.Pp[k, :] = A @ self.Pm[k, :] @ A.T + Q
        pass

    def _posterior_update(self, k: int, z_k):
        K = self.Pp[k] @ H.T @ np.linalg.inv(H @ self.Pp[k] @ H.T + R)
        self.xm[:, k] = self.xp[:, k] + K @ (z_k - H @ self.xp[:, k])
        self.Pm[k, :] = (np.eye(A.shape[0], A.shape[1]) - K @ H) @ self.Pp[k, :]
        pass

    def update(self, k, z_k):
        self._prior_update(k)
        self._posterior_update(k, z_k)

class SteadyStateKF:
    def __init__(self, T, x0) -> None:
        self.x_hat = np.zeros((x0.shape[0], T))
        P_inf = scipy.linalg.solve_discrete_are(A.T, H.T, Q, R)
        self.K_inf = P_inf @ H.T @ np.linalg.inv(H @ P_inf @ H.T + R)
        self.B = np.eye(A.shape[0], A.shape[1]) - self.K_inf @ H
        self.A = self.B @ A
        pass

    def update(self, k, z_k, u_k = np.zeros((16,))):
        self.x_hat[:, k] = self.A @ self.x_hat[:, k-1] + self.B @ u_k + self.K_inf @ z_k
        pass

if __name__ == "__main__":
    T = 100

    # loading data
    A = pd.read_csv("CubeModel_A.csv", header=None).values
    H = pd.read_csv("CubeModel_H.csv", header=None).values

    print("Is system stable? ", check_stability())
    print("Is system observable? ", check_observability())

    Q = np.eye(A.shape[0]) * 1e-6
    q_diag_one_measurement = [1e-6, 2e-5, 2e-5]
    q_diag = []
    for _ in range(6):
        q_diag += q_diag_one_measurement
    R = np.diag(q_diag)

    # init variables
    x0 = np.zeros((A.shape[0],))
    P0 = np.eye(A.shape[0], A.shape[1]) * 3e-4
    x = np.zeros((A.shape[0], T))
    x[:, 0] = np.squeeze(np.sqrt(P0) @ np.random.randn(A.shape[0], 1))
    z = np.zeros((H.shape[0], T))

    # Process noise variance: user input
    print("Choose process noise variance to run:")
    print(" (a) Q = 1e-6 I")
    print(" (b) Q = 1e-3 I")
    print(" (c) Q = 1e-9 I")
    print("Enter a, b, or c:")
    user_input = input()

    if user_input == "a":
        Q = 1e-06 * np.eye(A.shape[0])
    elif user_input == "b":
        Q = 1e-03 * np.eye(A.shape[0])
    elif user_input == "c":
        Q = 1e-09 * np.eye(A.shape[0])
    else:
        raise ValueError('Invalid input. Please choose either "a" or "b" or "c".')
    
    # init KF
    kf = KF(T, x0, P0)
    # init SS KF
    ss_kf = SteadyStateKF(T, x0)

    for k in range(1, T):
        simulate(k)
        ss_kf.update(k, z[:, k])
        kf.update(k, z[:, k])

    # estimation error
    error_kf = x - kf.xm
    error_ss_kf = x - ss_kf.x_hat
    
    # Select what states to plot (select 4).
    sel = [1, 2, 13, 14]

    fig = plt.figure()
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    # Figure 1: states and state estimates
    axs_est = subfigs[0].subplots(4, 1)
    subfigs[0].suptitle("States and state estimates (in deg or deg/s)")
    for i, ax in enumerate(axs_est):
        ax.plot(
            np.arange(0, T),
            x[sel[i] - 1, :] / np.pi * 180,
            label="true state",
            c="#2ca02c",
        )
        ax.plot(
            np.arange(0, T),
            kf.xm[sel[i] - 1, :] / np.pi * 180,
            label="TVKF estimate",
            c="#1f77b4",
        )
        ax.plot(
            np.arange(0, T),
            ss_kf.x_hat[sel[i] - 1, :] / np.pi * 180,
            label="SSKF estimate",
            c="#ff7f0e",
        )
        ax.set_ylabel(f"$x({sel[i]})$")
        ax.grid()
        if i == len(sel) - 1:
            ax.legend()
            ax.set_xlabel("Discrete-time step k")

    axs_err = subfigs[1].subplots(4, 1)
    subfigs[1].suptitle("Estimation error (in deg or deg/s)")
    for i, ax in enumerate(axs_err):
        ax.plot(
            np.arange(0, T),
            error_kf[sel[i] - 1, :] / np.pi * 180,
            label="TVKF estimate",
            c="#1f77b4",
        )
        ax.plot(
            np.arange(0, T),
            error_ss_kf[sel[i] - 1, :] / np.pi * 180,
            label="SSKF estimate",
            c="#ff7f0e",
        )
        ax.set_ylabel(f"$x({sel[i]})$")
        ax.grid()
        if i == len(sel) - 1:
            ax.legend()
            ax.set_xlabel("Discrete-time step k")

    # Analysis
    # Compute the poles of the error dynamics for the steady-state KF.
    poles, _ = np.linalg.eig(ss_kf.A)
    print("\n\n# Error Analysis:")
    print("Magnitude of error dynamic eigenvalues: ")
    print(np.abs(poles))

    # Compute squared estimation error.
    print("\n\n# Squared estimation error:")
    print(f"Time-varying KF: {np.sum(error_kf**2)}")
    print(f"Steady-state KF: {np.sum(error_ss_kf**2)}")

    plt.show()
    pass