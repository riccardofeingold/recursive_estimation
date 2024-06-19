import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Constants:
    H = 1
    R = 1
    P_0 = 1
    x_0 = 0    


def dynamics(k: int):
    if discrete:
        w = np.sign(np.random.randn())
    else:
        w = np.random.randn()
    z[k] = x_actual + w 
    pass

def update(k: int):
    K[k] = 1 / (1 + k)
    x_lin[k] = x_lin[k-1] + K[k] * (z[k] - x_lin[k-1])
    if discrete:
        x_lin_discrete[k] = 1 if x_lin[k] >= 0 else -1
        
        if np.abs(x_non_lin_discrete[k-1]) != 0:
            x_non_lin_discrete[k] = x_non_lin_discrete[k-1]
        else:
            if z[k] == 0:
                x_non_lin_discrete[k] = 0
            elif z[k] == 2:
                x_non_lin_discrete[k] = 1
            else:
                x_non_lin_discrete[k] = -1
    P[k] = 1 / (1 + k)
    pass

if __name__ == "__main__":
    HORIZON = 10000

    discrete: bool = int(input("Discrete or Continuous w(k)? 1 or 0: "))

    z = np.zeros((HORIZON, 1))
    K = np.zeros((HORIZON, 1))
    P = np.zeros((HORIZON, 1))
    x_lin = np.zeros((HORIZON, 1))
    x_lin_discrete = np.zeros((HORIZON, 1))
    x_non_lin_discrete = np.zeros((HORIZON, 1))

    if discrete:
        x_actual = np.sign(np.random.randn())
    else:
        x_actual = np.random.randn()
    
    # init
    x_lin[0] = 0
    x_lin_discrete[0] = 0
    x_non_lin_discrete[0] = 0
    K[0] = 1
    P[0] = 1

    for k in range(1, HORIZON):
        dynamics(k)
        update(k)
    
    plt.figure()
    ax = plt.axes()
    ax.scatter(np.linspace(0, HORIZON, HORIZON), x_lin, label="CRV: Linear Estimator")
    if discrete:
        ax.scatter(np.linspace(0, HORIZON, HORIZON), x_lin_discrete, label="DRV: Linear estimator", marker="^")
        ax.scatter(np.linspace(0, HORIZON, HORIZON), x_non_lin_discrete, label="DRV: Non Linear Estimator", marker="x")
    ax.plot(np.ones(HORIZON) * x_actual, color="red")
    ax.legend()
    plt.show()
    pass