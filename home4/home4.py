import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import random

@njit 
def step(n, M, directions):
    R = np.zeros(M)
    for i in range(M):
        point = np.zeros(2)
        for _ in range(n):
            direction = random.randint(0, 3)
            point += directions[direction]
        R[i] = np.sqrt(point[0]**2 + point[1]**2)
    return R

def main():
    M = 10000 #number simulations for each N
    N = np.array([10, 100, 500, 800]) 

    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    R = np.zeros((len(N), M))
    R_mean = np.zeros(len(N))

    for k, n in enumerate(N):
        R[k, :] = step(n, M, directions)
        R_mean[k] = np.sqrt(np.sum(R[k, :]**2) / M)

    print(np.polyfit(np.log(N), np.log(R_mean), 1)[0])

    fig, ax = plt.subplots()

    ax.loglog(N, R_mean, label="$\sqrt{<R^2(N)>}$")
    ax.loglog(N, 0.5*np.sqrt(N),'--', label="$1/2*N^{1/2}$")
    ax.set_xlabel("N")
    ax.set_ylabel("Distance")
    ax.set_title("Simulation")
    ax.legend()
    plt.savefig("plot.png")

if __name__ == "__main__":
    main()
