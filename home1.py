import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
#from methods import RK4

def step2(h,y,p):
    ky = h*dydt(p)
    kp = h*dpdt(y)
    return p+h*dpdt(y+ky/2), y+h*dydt(p+kp/2)

def step3(h,y,p):
    ky1 = h*dydt(p)
    kp1 = h*dpdt(y)
    ky2 = h*dydt(p+kp1/2)
    kp2 = h*dpdt(y+ky1/2)
    ky3 = h*dydt(p-kp1+2*kp2)
    kp3 = h*dpdt(y-ky1+2*ky2)

    return p+1/6*(kp1+4*kp2+kp3),y+1/6*(ky1+4*ky2+ky3)

def step4(h,y,p):
    k1y = dydt(p)
    k1p = dpdt(y)
    k2y = dydt(p+h*k1p/2)
    k2p = dpdt(y+h*k1y/2)
    k3y = dydt(p+h*k2p/2)
    k3p = dpdt(y+h*k2y/2)
    k4y = dydt(p+h*k3p)
    k4p = dpdt(y+h*k3y)

    return p+(h/6)*(k1p+2*(k2p+k3p)+k4p), y+(h/6)*(k1y+2*k2y+2*k3y+k4y)

def dpdt(y):
    return -4*(np.pi**2)*y

def dydt(p):
    return p

T = 2

hs = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.04])
errory2 = np.zeros(len(hs))
errory3 = np.zeros(len(hs))
errory4 = np.zeros(len(hs))

errorp2 = np.zeros(len(hs))
errorp3 = np.zeros(len(hs))
errorp4 = np.zeros(len(hs))

for j,h in enumerate(hs):
    antal = int(T/h)
    ts = np.linspace(0,T,antal+1)

    u2 = np.zeros((antal+1,2))
    u3 = np.zeros((antal+1,2))
    u4 = np.zeros((antal+1,2))
    u2[0,0], u3[0,0], u4[0,0] = 0,0,0
    u2[0,1], u3[0,1], u4[0,1] = 1,1,1

    yexact = np.cos(np.pi*2*ts)
    pexact = -2*np.pi*np.sin(np.pi*2*ts)

    for i in range(antal):
        u2[i+1,0],u2[i+1,1] = step2(h,u2[i,1],u2[i,0])
        u3[i+1,0],u3[i+1,1] = step3(h,u3[i,1],u3[i,0])
        u4[i+1,0],u4[i+1,1] = step4(h,u4[i,1],u4[i,0])

    errory2[j] = np.linalg.norm(u2[:,1]-yexact)
    errory3[j] = np.linalg.norm(u3[:,1]-yexact)
    errory4[j] = np.linalg.norm(u4[:,1]-yexact)

    errorp2[j] = np.linalg.norm(u2[:,0]-pexact)
    errorp3[j] = np.linalg.norm(u3[:,0]-pexact)
    errorp4[j] = np.linalg.norm(u4[:,0]-pexact)

slope = linregress(np.log(hs),np.log(errory4))

figy,ax_y = plt.subplots()
ax_y.plot(ts,u4[:,1],label="y4")
ax_y.plot(ts,yexact,'--',label="exact")
ax_y.legend()
ax_y.set_xlabel("t")
ax_y.set_ylabel("y")
figy.suptitle('Displacement depending on time')
figy.savefig("y4.png")

figp,ax_p = plt.subplots()
ax_p.plot(ts,u4[:,0],label="p4")
ax_p.plot(ts,pexact,'--',label="exact")
ax_p.legend()
ax_p.set_xlabel("t")
ax_p.set_ylabel("p")
figp.suptitle('Momentum depending on time')
figp.savefig("p4.png")

fig2,ax2 = plt.subplots()
ax2.loglog(hs,errory2,label="Second order")
ax2.loglog(hs,errory3,label="Third order")
ax2.loglog(hs,errory4,label="Fourth order")
ax2.loglog(hs,hs**3.5*4.5,label="test")
ax2.legend()
fig2.suptitle('L2-norm of error for displacement evaluated with different step sizes')
ax2.set_xlabel("h")
ax2.set_ylabel("Error")
fig2.savefig("errorsy.png")

fig3,ax3, = plt.subplots()
ax3.loglog(hs,errorp2,label="Second order")
ax3.loglog(hs,errorp3,label="Third order")
ax3.loglog(hs,errorp4,label="Fourth order")
ax3.legend()
fig3.suptitle('L2-norm of error for momentum evaluated with different step sizes')
ax3.set_xlabel("h")
ax3.set_ylabel("Error")
fig3.savefig("errorsp.png")