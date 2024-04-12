import matplotlib.pyplot as plt
import numpy as np
import random
from numba import njit

kb = 1.380649e-23
J = -kb
B = 0

@njit
def HeatBath(S, size, T):
    for _ in range(len(S)):
        for _ in range(len(S)):
            i = np.random.randint(0,size)
            j = np.random.randint(0,size)
            temp = np.exp(2*J/(kb*T)*(S[(i+1)%size, j] + S[(i-1)%size,j] + S[i, (j+1) % size] + S[i,(j-1)%size]))
            r = np.random.uniform(0,1)
            if r > temp/(1+temp):
                S[i,j] = 1
            else:
                S[i,j] = -1
    return S

@njit
def energy(S,size):
    E=0
    for i in range(size):
        for j in range(size):
            E+=Hamiltonian(S,size,i,j)
    return E/4/kb


@njit
def Metropolis(S, size, T):
    for _ in range(len(S)):
        for _ in range(len(S)):
            i = np.random.randint(0,size)
            j = np.random.randint(0,size)
            H1 = Hamiltonian(S,size,i,j)
            S[i,j] *= -1
            H2 = Hamiltonian(S,size,i,j)
            S[i,j] *= -1
            dE=H2-H1
            
            if dE >= 0:
                r = np.random.uniform(0,1)
                if r < np.exp(-dE/(kb*T)):
                    S[i,j] *= -1
            else:
                S[i,j] *= -1
                
    return S

@njit
def Hamiltonian(S,size,i,j):
    H = -S[i,j]*(J*(S[(i+1)%size, j] + S[(i-1)%size,j] + S[i, (j+1) % size] + S[i,(j-1)%size])+B)
    return H


def main():
    num_mc_calcs = 5
    size_vec = [8, 16, 32]
    timesteps = 500
    Ts = np.linspace(1.50,3.5,50)
    s = len(Ts)/10
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for size in size_vec:
        print(size)
        X_avg = np.zeros(len(Ts))
        Ms_avg = np.zeros(len(Ts))
        C_avg = np.zeros(len(Ts))
        four_cum_avg = np.zeros(len(Ts))


        N=(timesteps/s*size*size)

        for i in range(num_mc_calcs):
            Ms = np.zeros(len(Ts))
            Ms_square = np.zeros(len(Ts))
            X = np.zeros(len(Ts))
            E = np.zeros(len(Ts))
            E_square = np.zeros(len(Ts))
            C = np.zeros(len(Ts))
            four_cum = np.zeros(len(Ts))
            Ms_four = np.zeros(len(Ts))
            
            for l,T in enumerate(Ts):
                S = np.matrix([[random.choice([-1,1]) for _ in range(size)] for _ in range(size)])
                for _ in range(timesteps):
                    S = HeatBath(S,size,T)
                for i in range(timesteps):
                    S = HeatBath(S,size,T)
                    if i % s == 0:
                        Ms[l] += S.sum()
                        Ms_square[l] += S.sum()*S.sum()
                        E[l] += energy(S,size)
                        E_square[l] += energy(S,size)*energy(S,size)
                        Ms_four[l] += S.sum()/N**2*S.sum()*S.sum()*S.sum()



            X=(Ms_square/(timesteps/s*size*size)-1/((timesteps/s)**2*size*size)*Ms*Ms)/Ts
            Ms /= ((size*size)*timesteps/s)
            C = ((E_square/(timesteps/s*size*size)-1/((timesteps/s)**2*size*size)*E*E)/Ts)/Ts
            four_cum = (1 - Ms_four/(3*(Ms_square/(timesteps/s*size*size))**2))

            X_avg+=X
            Ms_avg+=np.abs(Ms)
            C_avg+=C
            four_cum_avg+=four_cum

        
        X_avg/=num_mc_calcs
        Ms_avg/=num_mc_calcs
        C_avg/=num_mc_calcs
        four_cum_avg/=num_mc_calcs
    
        ax.plot(Ts,Ms_avg,'.', label = f'L = {size}')
        ax2.plot(Ts,X_avg,'.', label = f'L = {size}')
        ax3.plot(Ts,C_avg,'.', label = f'L = {size}')
        ax4.plot(Ts,four_cum_avg,'.', label = f'L = {size}')
        
    # region Plotting
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    fig.suptitle('Order parameter')
    fig2.suptitle('Susceptibility')
    fig3.suptitle('Specific heat')
    fig4.suptitle('Fourth order cumulant')

    ax.set_xlabel('Temperature (K)')
    ax2.set_xlabel('Temperature (K)')
    ax3.set_xlabel('Temperature (K)')
    ax4.set_xlabel('Temperature (K)')

    ax.set_ylabel('$M$')
    ax2.set_ylabel('$\chi$')
    ax3.set_ylabel('$C_B$')
    ax4.set_ylabel('$U_L$')

    ax.figure.savefig("Magnetization.png")
    ax2.figure.savefig("Susceptibilty.png")
    ax3.figure.savefig("Specific_heat.png")
    ax4.figure.savefig("fourthCumulant.png")

    # endregion 
        
if __name__ == "__main__" :
    main()