import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def analytical(r,a):
    return ((1/(1-a**2))**2) * (np.e**((-a)*r) - np.e**(-r) * (1+(1/2)*(1-a**2)*r))

def S(r):
    return -(r*np.e**(-r))/(2)

def phi_min(r_min,a):
    return (1/(np.sqrt(2*a)))*(np.e**(a*r_min)-np.e**((-a)*r_min))

def phi_max(r_big,a):
    return -(1/np.sqrt(2*a))*np.e**((-a)*r_big)

@njit
def bode(yvec,h,N): #simplification of Bode's rule with equidistant points
    integral = 7*(yvec[0]+yvec[-1])+32*sum(yvec[1:N-1:2])+12*sum(yvec[2:N-2:4])+14*sum(yvec[4:N-4:4])
    integral *= (2*h)/45
    return integral

def solution(r,r_vec,h,N,a,i):
    term1 = S(r_vec[0:i])*phi_min(r_vec[0:i],a)
    term2 = S(r_vec[i:N])*phi_max(r_vec[i:N],a)
    integral = [bode(term1,h,N),bode(term2,h,N)]

    return phi_max(r,a)*integral[0]+phi_min(r,a)*integral[1]
      
    
def main():
    a = 4
    r_max = 30
    N = 1000
    h = r_max/N
    r_vec = np.linspace(0,r_max,N)
    sol = np.zeros(N)

    #error = np.zeros(N)
    analy = analytical(r_vec,a) 

    for i,r in enumerate(r_vec):
        sol[i] = solution(r,r_vec,h,N,a,i)

    sol[0] = 0 #BC

    fig, ax = plt.subplots()

    ax.plot(r_vec,sol,label="Numerical")
    ax.plot(r_vec,analy,"r--",label="Analytical")
    ax.set_xlabel("r")
    ax.set_ylabel("$\phi$")
    ax.legend()
    plt.savefig("solution.png")

if __name__ == "__main__":
    main()