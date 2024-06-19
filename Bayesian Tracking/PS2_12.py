# Problem Set2 - Problem 12
#
# Object on Circle - Recursive Filtering Algorithm
#
# Recursive Estimation
# Spring 2023
#
# --
# ETH Zurich
# Institute for Dynamic Systems and Control
#
# --
# Revision history
# [14.03.10, AS]    First version
# [08.03.23, WZ]    Python version


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def dynamics(k: int):
    x[k] = np.mod(x[k-1] + v[k], N)
    theta[k] = 2*np.pi * x[k] / N

    distance[k] = np.sqrt(
        (L - np.cos(theta[k]))**2 + np.sin(theta[k])**2
    ) + w[k]

def prior_update(k: int):
    for i in range(N):
        summation = 0
        for j in range(N):
            if i == np.mod(j + 1, N):
                summation += p * posterior[k-1, j]
            elif i == np.mod(j - 1, N):
                summation += (1 - p) * posterior[k-1, j]
            else:
                summation += 0
        prior[k, i] = summation
        pass
    pass

def measurement_update(k: int):
    for i in range(N):
        denominator = 0
        for j in range(N):
            theta = 2 * np.pi * j / N
            if np.abs(distance[k] - np.sqrt((L - np.cos(theta))**2 + np.sin(theta)**2)) <= e:
                denominator += 1/(2*e) * prior[k, j]
            else:
                denominator += 0
        
        theta = 2 * np.pi * i / N
        if np.abs(distance[k] - np.sqrt((L - np.cos(theta))**2 + np.sin(theta)**2)) <= e:
            numerator = 1/(2*e) * prior[k, i]
        else:
            numerator = 0
        
        posterior[k, i] = numerator / denominator
    pass

if __name__ == "__main__":
    answer = input("Which exercise do you want to test? a or b")
    
    N = 100
    MAX_ITERATION = 100
    x = np.zeros((MAX_ITERATION,))
    x[0] = N/4
    theta = np.zeros((MAX_ITERATION,))

    if answer == "a":
        e = 0.5
        system_params = [
            (2, 0.5),
            (2, 0.55),
            (0.1, 0.55),
            (0, 0.55)
        ]
    else:
        system_params = [
            (2, 0.45, 0.5),
            (2, 0.5, 0.5),
            (2, 0.9, 0.5),
            (2, 0.55, 0.9),
            (2, 0.55, 0.45),
        ]
        pass
    
    for i in range(len(system_params)):
        if len(system_params[0]) == 2:
            L, p = system_params[i]
        else:
            L, p, e = system_params[i]
        
        distance = np.zeros((MAX_ITERATION,))
        v = np.random.rand(MAX_ITERATION)
        v = np.where(v < p, 1, -1)
        w = e * (2 * np.random.rand(MAX_ITERATION) - 1)

        posterior = np.zeros((MAX_ITERATION, N))
        posterior[0, :] = 1/N # maximum ignorance, literally means that the model is not better than randomly guessing
        prior = np.zeros((MAX_ITERATION, N))

        for k in range(1, MAX_ITERATION):
            dynamics(k)
            prior_update(k)
            measurement_update(k)
        
        # Visualize the results
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlabel("POSITION $x(k)/N$ ")
        ax.set_ylabel("TIME STEP $k$")
        X = np.arange(0, N) / N
        Y = np.arange(0, MAX_ITERATION)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, posterior, rstride=1, cstride=1, cmap=cm.coolwarm)
        ax.plot3D(
            x / N,
            np.arange(0, MAX_ITERATION),
            np.ones((MAX_ITERATION,)) * np.max(posterior),
            label="Actual Position",
        )
        ax.legend()
        plt.show()
    pass