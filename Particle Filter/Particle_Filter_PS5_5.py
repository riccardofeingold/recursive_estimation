# Problem Set5 - Problem 5
#
# Particle Filter design for the terrain following problem
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
# [19.05.11, ST]    First version by Sebastian Trimpe for recitation
# [28.05.13, PR]    Adapted for Particle Filter Problem Set
# [08.03.23, WZ]    Python version


import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_altitude_and_gradient(x, y):
    """Altitude and gradient function

    Args:
        x (float): x-coordinates of person, in km
        y (float): y-coordinates of person, in km

    Returns:
        float: altitude evaluated at (x, y), in km
        np.ndarray: gradient evaluated at (x, y), shape (2,1)
    """
    coeff = np.array(
        [
            -1.394079911146759e1,
            4.318449838008606e1,
            -5.873035185058867e-2,
            -1.400961327221415e1,
            4.393262455274894e1,
            -6.301283314674033e-2,
            1.936241014498105e2,
            -5.881117127815230e2,
            -5.803913633249534e2,
            1.558126588975868e3,
            1.575320828668508e2,
            1.468258796264717e2,
            2.000000000000000e3,
        ]
    )

    def h(x, y):
        h = (
            1 * np.multiply(x**3, y**3)
            + coeff[0] * np.multiply(x**2, y**3)
            + coeff[1] * np.multiply(x, y**3)
            + coeff[2] * y**3
            + coeff[3] * np.multiply(x**3, y**2)
            + coeff[4] * np.multiply(x**3, y)
            + coeff[5] * x**3
            + coeff[6] * np.multiply(x**2, y**2)
            + coeff[7] * np.multiply(x**2, y)
            + coeff[8] * np.multiply(x, y**2)
            + coeff[9] * np.multiply(x, y)
            + coeff[10] * x
            + coeff[11] * y
            + coeff[12]
        )
        return h

    def dhdx(x, y):
        dhdx = (
            3 * np.multiply(x**2, y**3)
            + 2 * coeff[0] * np.multiply(x, y**3)
            + coeff[1] * y**3
            + 0 * coeff[2]
            + 3 * coeff[3] * np.multiply(x**2, y**2)
            + 3 * coeff[4] * np.multiply(x**2, y)
            + 3 * coeff[5] * x**2
            + 2 * coeff[6] * np.multiply(x, y**2)
            + 2 * coeff[7] * np.multiply(x, y)
            + coeff[8] * y**2
            + coeff[9] * y
            + coeff[10]
            + 0 * coeff[11]
            + 0 * coeff[12]
        )
        return dhdx

    def dhdy(x, y):
        dhdy = (
            3 * np.multiply(x**3, y**2)
            + 3 * coeff[0] * np.multiply(x**2, y**2)
            + 3 * coeff[1] * np.multiply(x, y**2)
            + 3 * coeff[2] * y**2
            + 2 * coeff[3] * np.multiply(x**3, y)
            + coeff[4] * x**3
            + 0 * coeff[5]
            + 2 * coeff[6] * np.multiply(x**2, y)
            + coeff[7] * x**2
            + 2 * coeff[8] * np.multiply(x, y)
            + coeff[9] * x
            + 0 * coeff[10]
            + coeff[11]
            + 0 * coeff[12]
        )
        return dhdy

    # convert from m to km
    altitude = 0.001 * h(x, y)
    gradient = 0.001 * np.array([dhdx(x, y), dhdy(x, y)])

    return altitude, gradient


def dynamics(k):
    """Person's motion

    Args:
        k (int): current time step
    """
    # TODO a): Simulation of person
    # Get current altitude and terrain gradient
    # grad = ...

    # Sample process noise
    # process_noise = ...

    # Simulate system dynamics
    # s(k+1) = s(k) - theta * grad / |grad| + v(k-1)
    # states[:, k] = ...

    # Sample measurement noise
    # measurement_noise = ...

    # Get current altitude of person:
    # altitude = ...

    # Save measurement
    # measurements[:, k] = ...

    pass


def prior_update(k):
    """Prior update

    Args:
        k (int): current time step

    Returns:
        np.ndarray: particles at time step k after prior update, shape (2, N)
    """
    # TODO b): Prior update
    # Get terrain gradient of particles
    # grads = ...

    # Draw noise samples
    # process_noise_samples = ...

    # Calculate prior particles
    priors = np.zeros((2, N))  # Modify this line

    return priors


def posteriori_update(k, priors):
    """Posteriori update

    Args:
        k (int): current time step
        priors (np.ndarray): particles at time step k after prior update, shape (2, N)
    """
    # TODO c): Posteriori update
    # Get measurement likelihood of particles
    # beta_i = f_w(z - h_i) with h_i the height of particle i, and z the given measurement
    # betas = ...

    # Build cumulative sum of particles (similar to a CDF)
    # beta_cum_sum = ...

    # Resample particles
    particles[k, :, :] = priors[:, :]  # Modify this line


def roughening(particles):
    """Roughening

    Args:
        particles (np.ndarray): particles after posteriori update, shape (2, N)

    Returns:
        np.ndarray: particles after roughening, shape (2, N)
    """
    # TODO d): Roughening
    # K = ...  # Roughening parameter
    # D = ...  # Dimension of the state space

    # Find maximal inter-sample variability
    # Ei = ...

    # Build diagonal matrix of standard deviations for drawing roughening samples:
    # StdRough = ...

    # Get roughening samples from Gaussian distribution with StdRough stand. dev.
    # deltaX = ...
    # and add them to posterior particles:
    # particles[:, :] = particles[:, :] + deltaX

    return particles


if __name__ == "__main__":
    # Simulation setup
    N = 300  # number of particles
    T = 90  # number of simulation steps

    # System parameters
    s0 = np.array([[7.5], [7.5]])
    P0 = 1 / 25 * np.eye(2)
    theta = 1 / 12
    Q = (theta / 4) ** 2 * np.eye(2)
    R = (0.05) ** 2

    # Storage arrays
    states = np.zeros((2, T + 1))
    measurements = np.zeros((1, T + 1))
    particles = np.zeros((T + 1, 2, N))

    # Draw random initial state
    states[:, 0:1] = s0 + np.sqrt(P0) @ np.random.randn(2, 1)

    for k in range(1, T + 1):
        dynamics(k)

    # Particle initialization
    # TODO b): PF initialization
    # particles[0, :, :] = ...

    for k in range(1, T + 1):
        # Prior update
        priors = prior_update(k)
        # Posteriori update
        posteriori_update(k, priors)
        # Roughening
        particles[k, :, :] = roughening(particles[k, :, :])

    ##############################
    # Following code needs no changes
    ##############################

    # Animation setup
    pause_msec = 10  # delay between time steps, in milliseconds

    # Plot Results
    step = 0.2  # grid resolution
    gridding = np.arange(0, 10 + step, step)

    # Axis limits
    x_lim = np.array([0, gridding[-1]])
    y_lim = np.array([0, gridding[-1]])
    z_lim = np.array([-1, 5])

    # 3D plot
    fig = plt.figure(figsize=(13, 7))
    fig.tight_layout()

    x_mesh, y_mesh = np.meshgrid(gridding, gridding)
    h_mesh, _ = get_altitude_and_gradient(x_mesh, y_mesh)

    ax_left = fig.add_subplot(121, projection="3d")
    ax_left.view_init(elev=55, azim=225, roll=0)
    ax_left.plot_surface(x_mesh, y_mesh, h_mesh, cmap=cm.coolwarm, alpha=0.7)
    ax_left.set_title("Step: k=0  Paricles: N=0")
    ax_left.set_xlabel("Location $x$ (km)")
    ax_left.set_ylabel("Location $y$ (km)")
    ax_left.set_zlabel("Altitude $a$ (km)")

    ax_right = fig.add_subplot(122)
    ax_right.contourf(x_mesh, y_mesh, h_mesh)
    ax_right.set_title("Level-Curves Plot")
    ax_right.set_xlabel("Location $x$ (km)")
    ax_right.set_ylabel("Location $y$ (km)")

    # Plot particles
    if np.any(particles[-1, :, :]):
        h, _ = get_altitude_and_gradient(particles[0, 0, :], particles[0, 1, :])
        p_left = ax_left.scatter(particles[0, 0, :], particles[0, 1, :], h, c="g")
        p_right = ax_right.scatter(
            particles[0, 0, :], particles[0, 1, :], label="Estimates"
        )
    # Plot true state
    h, _ = get_altitude_and_gradient(states[0, 0], states[1, 0])
    p_left_true = ax_left.plot(states[0, 0], states[1, 0], h, "ro")[0]
    p_right_true = ax_right.plot(states[0, 0], states[1, 0], "ro", label="True state")[
        0
    ]

    # Plot particles
    def animate(k):
        if k > T:
            return
        # Update particles
        if np.any(particles[-1, :, :]):  # check if prior update is implemented
            h, _ = get_altitude_and_gradient(particles[k, 0, :], particles[k, 1, :])
            p_left._offsets3d = (particles[k, 0, :], particles[k, 1, :], h)
            p_right.set_offsets(particles[k, :, :].T)
            ax_left.set_title(f"Step: k={k}  Paricles: N={N}")
        else:
            ax_left.set_title(f"Step: k={k}  Paricles: N=0")

        # Update true state
        h, _ = get_altitude_and_gradient(states[0, :k], states[1, :k])
        p_left_true.set_data(states[:2, :k])
        p_left_true.set_3d_properties(h)
        h, _ = get_altitude_and_gradient(states[0, k], states[1, k])
        p_right_true.set_xdata(states[0, :k])
        p_right_true.set_ydata(states[1, :k])

    ani = animation.FuncAnimation(
        fig, animate, interval=pause_msec, cache_frame_data=False
    )
    plt.legend()
    plt.show()
