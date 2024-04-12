import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def integrand1(b,p):
    return 4*b*p/((p*p+b)**2*(np.sqrt(1-b*b/(p*p+b)**2)))

def integrand2(p,b,U,E,rmin):
    return -4*b*p/(((p*p+rmin)**2)*np.sqrt(1-b*b/(p*p+rmin)**2-U/E))

def analytical1(b,rmax):
    return np.pi-2*np.arcsin(b/rmax)

def analytical2(b,rmax,U,E):
    return 2*np.arcsin(b/(rmax*np.sqrt(1-U/E)))-2*np.arcsin(b/rmax)

def main():
    rmax = 3
    E = 6
    U = -1
    bs = np.linspace(0,rmax,101)
    rmins = bs/(np.sqrt(1-U/E))
    solN1 = np.zeros(len(bs))
    solN1[0] = np.pi

    solN2 = np.zeros(len(bs))
    
    for i,b in enumerate(bs[1:-1]):
        solN1[i+1] = quad(lambda p: integrand1(b,p), 0, np.sqrt(rmax-b))[0]
        solN2[i+1] = solN1[i+1] + quad(lambda p: integrand2(p,b,U,E,rmins[i+1]), 0, np.sqrt(rmax-rmins[i+1]))[0]
    
    solA1 = analytical1(bs, rmax)
    solA2 = analytical2(bs[:-1], rmax, U, E)

    fig, ax = plt.subplots()
    ax.plot(bs, solN1, label = 'Numerical')
    ax.plot(bs, solA1, '--', label = 'Analytical')
    ax.set_ylabel("$\Theta$ (rad)")
    ax.set_xlabel("b")
    ax.legend()
    ax.set_title("Case $E<U_0$")
    plt.savefig("plot1.png")

    fig1, ax1 = plt.subplots()
    ax1.plot(bs[:-1], solN2[:-1], label = 'Numerical')
    ax1.plot(bs[:-1], solA2, '--', label = 'Analytical')
    ax1.set_ylabel("$\Theta$ (rad)")
    ax1.set_xlabel("b")
    ax1.legend()
    ax1.set_title("Case $E>U_0$")
    plt.savefig("plot2.png")

if __name__ == "__main__":
    main()