import numpy as np
import scipy
import matplotlib.pyplot as plt
from dataclasses import dataclass

import scipy.linalg

@dataclass
class SystemDynamics:
    alpha1: float = 0.1
    alpha2: float = 0.5
    alpha3: float = 0.2

    A: np.dtype = np.array([
        [1 - alpha1, 0, 0],
        [0, 1, 0],
        [alpha1, 0, 1 - alpha3]
    ])
    B: np.dtype = np.array([
        0.5, 0.5, 0
    ]).T
    H: np.dtype = np.array([
        [0, 1, 0],
        [0, 0, 1]
    ])

    x0: np.dtype = np.ones((3,))* 5
    P0: np.dtype = np.eye(3, 3)
    Q: np.dtype = np.array([
        [1/40, 0, 0],
        [0, 1/10, 0],
        [0, 0, 1/5]
    ])
    R: np.dtype = np.eye(2, 2) * 0.5


class KalmanFilter:
    def __init__(self, horizon):
        self.K = np.zeros((horizon, 3, 2))
        self.xp = np.zeros((horizon, 3,))
        self.Pp = np.zeros((horizon, 3, 3))
        self.xm = np.zeros((horizon, 3,))
        self.Pm = np.zeros((horizon, 3, 3))
        pass

    def prior_update(self, k: int, u_k: float):
        self.xp[k] = SystemDynamics.A @ self.xm[k-1] + SystemDynamics.B * u_k
        self.Pp[k] = SystemDynamics.A @ self.Pm[k-1] @ SystemDynamics.A.T + SystemDynamics.Q
        pass
    
    def posterior_update(self, k: int, z_k: np.dtype):
        self.K[k] = self.Pp[k] @ SystemDynamics.H.T @ np.linalg.inv(SystemDynamics.H @ self.Pp[k] @ SystemDynamics.H.T + SystemDynamics.R)
        self.xm[k] = self.xp[k] + self.K[k] @ (z_k - SystemDynamics.H @ self.xp[k])
        self.Pm[k] = (np.eye(3, 3) - self.K[k] @ SystemDynamics.H) @ self.Pp[k]
        pass

class Simulation:
    def __init__(self, horizon) -> None:
        self.x = np.zeros((horizon, 3,))
        self.x[0] = SystemDynamics.x0
        self.z = np.zeros((horizon, 2,))
        pass

    def simulate(self, k: int, u_k: float):
        v = np.random.multivariate_normal([0, 0, 0], SystemDynamics.Q)
        w = np.random.multivariate_normal([0, 0], SystemDynamics.R)
        self.x[k] = SystemDynamics.A @ self.x[k-1] + SystemDynamics.B * u_k + v
        self.z[k] = SystemDynamics.H @ self.x[k] + w
        
        return self.z[k]

def compute_p_inf() -> np.dtype:
    return scipy.linalg.solve_discrete_are(a=SystemDynamics.A.T, b=SystemDynamics.H.T, q=SystemDynamics.Q, r=SystemDynamics.R)

if __name__ == "__main__":
    NUM_ITERATIONS = 1000
    NO_RANDOM = False
    
    # Choose which part of the problem you would like to run:
    print("Choose part of the problem to run: ")
    PART = input()

    if PART == "d":
        u = 5 * np.abs(np.sin(np.arange(0, NUM_ITERATIONS)))
    else:
        u = 5 * np.ones((NUM_ITERATIONS,))

    if PART == "e":
        SystemDynamics.H = np.array([
            [1, 0, 0],
            [0, 0, 1]
        ])
    elif PART == "f" or PART == "g":
        SystemDynamics.A = np.array([
            [1 - SystemDynamics.alpha1, 0, 0],
            [0, 1 - SystemDynamics.alpha2, 0],
            [SystemDynamics.alpha1, SystemDynamics.alpha2, 1 - SystemDynamics.alpha3]
        ])
        SystemDynamics.H = np.array([
            [1, 0, 0],
            [0, 0, 1]
        ])
        P_inf = compute_p_inf()
        print("P_inf is equal to: ", P_inf)
    else:
        P_inf = compute_p_inf()
        print("P_inf is equal to: ", P_inf)

    sim = Simulation(NUM_ITERATIONS)
    kf = KalmanFilter(NUM_ITERATIONS)
    kf.xm[0] = SystemDynamics.x0
    kf.Pm[0] = SystemDynamics.P0 if NO_RANDOM else np.diag(5 * np.random.rand(3)) @ SystemDynamics.P0
    u_k = 5

    for k in range(1, NUM_ITERATIONS):
        if PART == "g":
            SystemDynamics.A[1, 1] = 0.5 if k % 3 == 0 else 0
            SystemDynamics.A[2, 1] = 0.5 if k % 3 == 0 else 0
        z_k = sim.simulate(k, u[k])
        kf.prior_update(k, u[k])
        kf.posterior_update(k, z_k)
        pass
    
    print("Pp converged to ", kf.Pp[-1])
    
     # Plot results
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    # Mean
    axs_mean = subfigs[0].subplots(3, 1)
    for i, ax in enumerate(axs_mean):
        ax.plot(np.arange(0, NUM_ITERATIONS), sim.x[:, i], "k-", label="True state")
        ax.plot(
            np.arange(0, NUM_ITERATIONS),
            kf.xm[:, i] + np.sqrt(kf.Pm[:, i, i]),
            "r--",
            label="+/- 1 standard deviation",
        )
        ax.plot(np.arange(0, NUM_ITERATIONS), kf.xm[:, i], "b--", label="Estimated state")
        ax.plot(np.arange(0, NUM_ITERATIONS), kf.xm[:, i] - np.sqrt(kf.Pm[:, i, i]), "r--")
        ax.set_xlabel("Time step k")
        ax.set_ylabel(f"Tank {i+1} Level, x({i+1})")
        ax.legend()

    # Variances
    ax_var = subfigs[1].subplots(1, 1)
    for row in range(3):
        for col in range(3):
            if col < row:
                continue
            subscript = str(row + 1) + str(col + 1)
            ax_var.plot(
                np.arange(1, NUM_ITERATIONS), kf.Pp[1:, row, col], label="$P_{" + subscript + "}$"
            )

    ax_var.set_xlabel("Time step k")
    ax_var.set_ylabel("Covariance matrix entry value")
    ax_var.legend()

    plt.show()