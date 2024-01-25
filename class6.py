import numpy as np
from methods import RK2,RK3,RK4
import matplotlib.pyplot as plt

def f(x,y):
	return -x*y


T = 5
h = 0.5
antal = int(T/h)
ts = np.linspace(0,T,int(antal))
u2 = np.zeros(len(ts))
u2[0] = 1
u3 = np.zeros(len(ts))
u3[0] = 1
u4 = np.zeros(len(ts))
u4[0] = 1
exact = np.exp(-ts**2/2)

for i,t in enumerate(np.delete(ts,antal-1)):
	u2[i+1] = RK2.step(h,f,t,u2[i])
	u3[i+1] = RK3.step(h,f,t,u3[i])
	u4[i+1] = RK4.step(h,f,t,u4[i])

fig,ax = plt.subplots()
labels = ["2:nd order","3:rd order","4:th order","exact"]
ax.plot(ts,u2,label = labels[0])
ax.plot(ts,u3,label = labels[1])
ax.plot(ts,u4,label = labels[2])
ax.plot(ts,exact,label = labels[3])

print(np.linalg.norm(u2)-np.linalg.norm(exact))
print(np.linalg.norm(u3)-np.linalg.norm(exact))
print(np.linalg.norm(u4)-np.linalg.norm(exact))

ax.legend()
plt.savefig('class6.png')
plt.show()