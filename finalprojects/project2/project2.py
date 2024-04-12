import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from numba import njit

@njit
def V(V0,a,r):
    return 4*V0*((a/r)**12-(a/r)**6)

@njit
def foo(r,b,E,V0,a):
    return 1-b*b/(r*r)-V(V0,a,r)/E

@njit
def integrand1(b,p):
    return 4*b*p/((p*p+b)**2*(np.sqrt(1-b*b/(p*p+b)**2)))

@njit
def integrand2(p,b,E,V0,a,rmin):
    return -4*b*p/(((p*p+rmin)**2)*np.sqrt(foo(p*p+rmin,b,E,V0,a)))

def main():
    V0 = 1
    rmax = 3
    a = rmax/3
    bs = [0.5, 0.8, 1, 1.3]
    Es = np.linspace(0.01*V0,100*V0,1000)
    solE = np.zeros(len(Es))
    d_Cs = np.zeros(len(bs))

    fig, ax = plt.subplots()
    ax.set_ylabel("$\Theta$ (rad)")
    ax.set_xlabel("E")
    

    for b in bs:
        integral = quad(lambda p: integrand1(b,p), 0.001, np.sqrt(rmax-b))[0] #only need to calculate the first integral once for each b, doesn't depend on E
        for i,E in enumerate(Es): #calculate varying E
            rmin = fsolve(lambda r: foo(r,b,E,V0,a), 0.5)
            solE[i] = integral + quad(lambda p: integrand2(p,b,E,V0,a,rmin), 0.001, np.sqrt(rmax-rmin))[0]

    
        ax.plot(Es, solE, label = f'b = {b}')
    
    ax.legend()
    fig.savefig("plotE.png")

    fig1, ax1 = plt.subplots()
    ax1.set_ylabel("$\Theta$ (rad)")
    ax1.set_xlabel("b")
    ax1.set_title(f"Solution")

    fig2, ax2 = plt.subplots(2,2)
    fig2.suptitle('Differential cross section')
    fig2.tight_layout()

    Es = [0.1, 1, 5, 100]
    bs = np.linspace(0.01,rmax-0.7,1000)
    solb = np.zeros(len(bs))

    for j,E in enumerate(Es):
        for i,b in enumerate(bs): #calculate by varying b
            rmin = fsolve(lambda r: foo(r,b,E,V0,a), 0.5)
            solb[i] = quad(lambda p: integrand1(b,p), 0.001, np.sqrt(rmax-b))[0] + quad(lambda p: integrand2(p,b,E,V0,a,rmin), 0.001, np.sqrt(rmax-rmin))[0]

        d_Cs = bs[:-1]/np.sin(solb)[:-1]*abs((bs[1]-bs[0])/np.diff(solb)) #calculate differential cross section
        ax1.plot(bs, solb, label = f'E = {E}')
        ax2[j//2, j%2].plot(bs[:-1], d_Cs, label = f'E = {E}')
        # ax2[j//2, j%2].set_ylabel("$\\frac{d\sigma}{d\Omega}$")
        # ax2[j//2, j%2].set_xlabel("b")
        ax2[j//2, j%2].set_xbound(0,3)
        ax2[j//2, j%2].set_ybound(-10,10)
        ax2[j//2, j%2].set_title(f'E = {E}')
        
    ax1.legend()
    fig1.savefig("plotb.png")
    fig2.savefig("plotdiff.png")
    

    

if __name__ == "__main__":
    main()